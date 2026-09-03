#!/usr/bin/env python3
"""超市分拣比赛裁判系统（流程化计分）。

以"流程 (cycle)"为单位打分：一个流程 = 把一个目标物体从货架搬到配送台的完整
循环，含 5 个有序步骤 S1..S5，必须按序完成、不能跳步；只有走完 S5 才结算一次分。
碰撞/碰倒/掉落等扣分只在该流程窗口内评估，从这个流程的得分里扣。

裁判用 MuJoCo 地面真值判定，由 server 每物理步调用 update(mj_data)。
"""
import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULTS = {
    "time_limit_s": 420.0,
    "scores": {"s1": 1, "s2": 2, "s3": 6, "s4": 1, "s5": 10, "upright_bonus": 5},
    "penalties": {"collision": 5, "topple": 5, "drop": 3},
    "thresholds": {
        "carry_out_dist": 0.20,   # S3: 目标离初始货位水平位移阈值(撤出货架)
        "settle_speed": 0.02,     # S5: 静止判定线速度 m/s
        "upright_tol_deg": 15.0,  # B1: 直立加分的最大倾角
        "topple_tilt_deg": 30.0,  # C2: 碰倒的倾角阈值
        "topple_xy_shift": 0.05,  # C2: 碰倒的水平位移阈值 m
        "drop_z": 0.30,           # C3: 掉落判定高度 m
    },
    "zones": {
        "picking": {"x": [-2.5, 2.5], "y": [1.70, 3.25]},
        # S4 base 到达判定：机器人常站在配送区北缘外伸手放置，故向北放宽 ~0.35m
        "delivery_base": {"x": [-2.420, -1.460], "y": [-3.880, -2.620]},
        # S5 物体判定框：严格对应桌面上方区域
        "delivery_box": {"x": [-2.420, -1.460], "y": [-3.630, -3.190], "z": [0.74, 1.05]},
    },
    "robot_links": [
        "agv_link", "slide_link",
        "rgt_arm_link1", "rgt_arm_link2", "rgt_arm_link3",
        "rgt_arm_link4", "rgt_arm_link5", "rgt_arm_link6",
    ],
    "grasp_fingers": ["rgt_finger_left_link", "rgt_finger_right_link"],
    "touch_links": ["rgt_finger_left_link", "rgt_finger_right_link", "rgt_arm_link6"],
    "structures": [
        "shelf_A", "shelf_B", "shelf_C", "shelf_D", "shelf_E",
        "perimeter_walls", "delivery_table", "dynamic_obstacle_corridor",
        "dynamic_obstacle_box_01", "dynamic_obstacle_box_02", "dynamic_obstacle_box_03",
        "dynamic_obstacle_box_04", "dynamic_obstacle_box_05",
    ],
}


def _merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


class Flow:
    """单个流程的状态。step: 已达成的步号(0..5)。"""
    def __init__(self, t0):
        self.step = 1            # S1 到达取货区(开启时即达成)
        self.target = None       # 绑定的目标 body(S3 时确定)
        self.touched = set()     # 已 touch 的目标
        self.collided = False
        self.toppled = False
        self.dropped = False
        self.t = {"s1": t0, "s2": None, "s3": None, "s4": None, "s5": None}
        self.upright = False


class Referee:
    def __init__(self, mj_model, target_bodies, object_bodies, config=None):
        self.mj_model = mj_model
        self.time_stamp = time.time()
        if isinstance(config, str) and config and os.path.exists(config):
            config = json.load(open(config, "r", encoding="utf-8"))
        self.cfg = _merge(DEFAULTS, config if isinstance(config, dict) else None)

        self.targets = list(target_bodies)
        self.objects = list(object_bodies)
        self.non_targets = [b for b in self.objects if b not in self.targets]

        # body name -> id（缺失的名字静默跳过，兼容不同场景）
        self.bid = {}
        for name in set(self.cfg["robot_links"] + self.cfg["touch_links"]
                        + self.cfg["grasp_fingers"] + self.cfg["structures"] + self.objects):
            try:
                self.bid[name] = int(mj_model.body(name).id)
            except KeyError:
                pass

        self.reset()

    def reset(self, mj_data=None):
        """Start a fresh scoring run while keeping the same targets/config."""
        self.time_stamp = time.time()
        self.init_pos = {}     # 首帧快照(碰倒/撤出货架的基线)
        self.snapped = False
        self.finished = False

        self.flow = None
        self.scored = set()          # 已成功计分的目标
        self.records = []            # 每个结算流程的明细
        self.events = []             # 文本事件流
        self._info = ""
        if mj_data is not None:
            self._snapshot(mj_data)
            self._build_info(float(mj_data.time))

    # ---------- 几何/接触工具 ----------
    def _snapshot(self, mj_data):
        for b in self.objects:
            if b in self.bid:
                self.init_pos[b] = mj_data.body(b).xpos.copy()
        self.snapped = True

    def _base_xy(self, mj_data):
        return mj_data.site("base_link").xpos[:2]

    @staticmethod
    def _in(zone, p):
        ok = zone["x"][0] <= p[0] <= zone["x"][1] and zone["y"][0] <= p[1] <= zone["y"][1]
        if "z" in zone:
            ok = ok and zone["z"][0] <= p[2] <= zone["z"][1]
        return ok

    def _contact_pairs(self, mj_data):
        """本 tick 处于接触的 (bodyid, bodyid) 对集合(双向)。"""
        pairs = set()
        gb = self.mj_model.geom_bodyid
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            b1, b2 = int(gb[c.geom1]), int(gb[c.geom2])
            pairs.add((b1, b2))
            pairs.add((b2, b1))
        return pairs

    def _touch(self, pairs, links, obj):
        oid = self.bid.get(obj)
        if oid is None:
            return False
        return any((self.bid[l], oid) in pairs for l in links if l in self.bid)

    def _gripped(self, pairs, obj):
        """两指同时接触 = 夹持。"""
        return all(self._touch(pairs, [f], obj) for f in self.cfg["grasp_fingers"])

    def _pos(self, mj_data, obj):
        return mj_data.body(obj).xpos

    def _tilt_deg(self, mj_data, obj):
        R = Rotation.from_quat(mj_data.body(obj).xquat[[1, 2, 3, 0]]).as_matrix()
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))

    def _speed(self, mj_data, obj):
        jid = int(self.mj_model.body(obj).jntadr[0])
        dof = int(self.mj_model.jnt_dofadr[jid])
        return float(np.linalg.norm(mj_data.qvel[dof:dof + 3]))

    def _xy_shift(self, mj_data, obj):
        return float(np.linalg.norm(self._pos(mj_data, obj)[:2] - self.init_pos[obj][:2]))

    # ---------- 主循环 ----------
    def update(self, mj_data):
        if self.finished:
            return
        if not self.snapped:
            self._snapshot(mj_data)
        t = float(mj_data.time)

        if t >= self.cfg["time_limit_s"]:
            self._finalize(t, reason="time_limit")
            return

        pairs = self._contact_pairs(mj_data)
        base_xy = self._base_xy(mj_data)
        in_pick = self._in(self.cfg["zones"]["picking"], base_xy)
        in_deliv = self._in(self.cfg["zones"]["delivery_base"], base_xy)
        remaining = [b for b in self.targets if b not in self.scored]

        # 开启新流程：手上没有在搬、进入取货区、还有目标
        if self.flow is None and remaining and in_pick:
            self.flow = Flow(t)
            self._log(t, "S1 到达取货区，开启新流程")

        f = self.flow
        if f is None:
            self._build_info(t)
            return

        # --- 流程窗口内的扣分监控 ---
        if not f.collided and self._robot_hits_structure(pairs):
            f.collided = True
            self._log(t, "C1 撞到结构 (−%d)" % self.cfg["penalties"]["collision"])
        if not f.toppled:
            hit = self._toppled_other_object(mj_data, f.target)
            if hit:
                f.toppled = True
                self._log(t, "C2 碰倒其他商品 %s (−%d)" % (hit, self.cfg["penalties"]["topple"]))

        thr = self.cfg["thresholds"]

        # --- 掉落判定(S3 之后、S5 之前) ---
        if f.step in (3, 4) and f.target is not None:
            if (not self._gripped(pairs, f.target)) and self._pos(mj_data, f.target)[2] < thr["drop_z"]:
                f.dropped = True
                self._settle_flow(t, completed=False)
                return

        # --- 按序推进 S2..S5(不能跳步) ---
        if f.step == 1:  # 等 S2 touch
            for tgt in remaining:
                if self._touch(pairs, self.cfg["touch_links"], tgt):
                    f.touched.add(tgt)
            if f.touched:
                f.step = 2
                f.t["s2"] = t
                self._log(t, "S2 够到目标")
        elif f.step == 2:  # 等 S3 抓取撤出货架
            for tgt in [b for b in f.touched if b in remaining]:
                if self._gripped(pairs, tgt) and self._xy_shift(mj_data, tgt) >= thr["carry_out_dist"]:
                    f.target = tgt
                    f.step = 3
                    f.t["s3"] = t
                    self._log(t, "S3 抓取并撤出货架，绑定目标 %s" % tgt)
                    break
        elif f.step == 3:  # 等 S4 携物到配送区
            if in_deliv and self._gripped(pairs, f.target):
                f.step = 4
                f.t["s4"] = t
                self._log(t, "S4 携物到达配送区")
        elif f.step == 4:  # 等 S5 准确放置
            in_box = self._in(self.cfg["zones"]["delivery_box"], self._pos(mj_data, f.target))
            if in_box and (not self._gripped(pairs, f.target)) and self._speed(mj_data, f.target) < thr["settle_speed"]:
                f.upright = self._tilt_deg(mj_data, f.target) <= thr["upright_tol_deg"]
                f.t["s5"] = t
                self._settle_flow(t, completed=True)
                return

        self._build_info(t)

    def _robot_hits_structure(self, pairs):
        for l in self.cfg["robot_links"]:
            lid = self.bid.get(l)
            if lid is None:
                continue
            for s in self.cfg["structures"]:
                sid = self.bid.get(s)
                if sid is not None and (lid, sid) in pairs:
                    return True
        return False

    def _toppled_other_object(self, mj_data, carried_target=None):
        thr = self.cfg["thresholds"]
        for b in self.objects:
            if carried_target is not None and b == carried_target:
                continue
            if b not in self.init_pos:
                continue
            if self._tilt_deg(mj_data, b) > thr["topple_tilt_deg"] or self._xy_shift(mj_data, b) > thr["topple_xy_shift"]:
                return b
        return None

    def _settle_flow(self, t, completed):
        """结算当前流程：记录明细、累计分数、开放下一流程。"""
        f = self.flow
        pen = self.cfg["penalties"]
        deductions = (pen["collision"] if f.collided else 0) + (pen["topple"] if f.toppled else 0)
        if completed:
            base = sum(self.cfg["scores"][s] for s in ("s1", "s2", "s3", "s4", "s5"))
            bonus = self.cfg["scores"]["upright_bonus"] if f.upright else 0
            net = base + bonus - deductions
            self.scored.add(f.target)
            self._log(t, "S5 准确放置%s，流程结算 净分 %+d（完成 %d/%d）"
                      % ("(直立+%d)" % self.cfg["scores"]["upright_bonus"] if f.upright else "",
                         net, len(self.scored), len(self.targets)))
        else:
            net = -(pen["drop"]) - deductions
            self._log(t, "C3 搬运掉落，流程作废 净分 %+d（目标 %s 可重试）" % (net, f.target))
        self.records.append({
            "target": f.target,
            "completed": bool(completed),
            "upright": bool(f.upright),
            "collided": bool(f.collided),
            "toppled": bool(f.toppled),
            "dropped": bool(f.dropped),
            "net": int(net),
            "steps": dict(f.t),
        })
        self.flow = None

    def _finalize(self, t, reason):
        if self.flow is not None:   # 进行中的流程不计分
            self._log(t, "时限到，进行中的流程未完成，不计分")
            self.flow = None
        self.finished = True
        self._log(t, "本局结束(%s) 总分 %d，完成 %d/%d 个目标"
                  % (reason, self.total_score, len(self.scored), len(self.targets)))
        self._build_info(t)

    # ---------- 输出 ----------
    @property
    def total_score(self):
        return int(sum(r["net"] for r in self.records))

    @property
    def completed_count(self):
        return len(self.scored)

    def _log(self, t, msg):
        line = ">>>>>> %6.2fs: %s" % (t, msg)
        print(line)
        self.events.append(line)

    def _build_info(self, t):
        step = self.flow.step if self.flow else 0
        self._info = ("t=%.1fs score=%d done=%d/%d flow_step=S%d"
                      % (t, self.total_score, self.completed_count, len(self.targets), step))

    @property
    def game_info(self):
        return self._info

    def result_dict(self):
        return {
            "total_score": self.total_score,
            "completed": self.completed_count,
            "target_count": len(self.targets),
            "time_limit_s": self.cfg["time_limit_s"],
            "total_time_s": max([r["steps"]["s5"] for r in self.records
                                 if r["completed"] and r["steps"]["s5"] is not None], default=None),
            "targets": self.targets,
            "flows": self.records,
        }

    def save_results(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(
                os.path.dirname(__file__),
                "referee_results_%s.json" % time.strftime("%Y%m%d-%H%M%S", time.localtime(self.time_stamp)))
        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(self.result_dict(), fp, indent=2, ensure_ascii=False)
        print("[referee] results saved to %s" % file_path)
        return file_path
