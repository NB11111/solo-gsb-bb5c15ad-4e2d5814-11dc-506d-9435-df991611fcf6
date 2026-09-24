"""solver 单元测试：固定用例 + 小规模暴力枚举交叉验证。"""

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.solver import (  # noqa: E402
    CapacityExceededError,
    TargetUnreachableError,
    Ware,
    min_capacity_for_target,
    select_within_kiln_limit,
    solve,
)


def _feasible(items_by_id, group, capacity):
    total = sum(items_by_id[i].volume for i in group)
    if total > capacity:
        return False
    lo = max(items_by_id[i].temp_low for i in group)
    hi = min(items_by_id[i].temp_high for i in group)
    return lo <= hi


def _brute_force(items, capacity):
    """枚举所有集合划分，返回 (最少窑次, 字典序最小方案)。"""
    ids = sorted(i.item_id for i in items)
    by_id = {i.item_id: i for i in items}
    best = None  # (窑次数, 方案元组)

    def partitions(s, groups):
        nonlocal best
        if not s:
            plan = tuple(sorted(tuple(sorted(g)) for g in groups))
            k = len(plan)
            if best is None or (k, plan) < (best[0], best[1]):
                best = (k, plan)
            return
        elem = s[0]
        rest = s[1:]
        # elem 放入已有某一组
        for gi in range(len(groups)):
            ng = groups[gi] | {elem}
            if _feasible(by_id, ng, capacity):
                new_groups = groups[:gi] + [ng] + groups[gi + 1:]
                partitions(rest, new_groups)
        # elem 单开一组
        if _feasible(by_id, {elem}, capacity):
            partitions(rest, groups + [{elem}])

    partitions(ids, [])
    return best


def _run(items, capacity):
    kilns = solve(items, capacity)
    plan = tuple(k.item_ids for k in kilns)
    return len(kilns), plan


def test_overcapacity_single_item():
    items = [Ware("a", 11, 100, 200)]
    try:
        solve(items, 10)
        assert False, "应抛出 CapacityExceededError"
    except CapacityExceededError:
        pass


def test_basic_capacity_split():
    # 每件 6，容量 10 -> 必须 3 窑
    items = [Ware(str(i), 6, 100, 200) for i in range(1, 4)]
    k, plan = _run(items, 10)
    assert k == 3
    assert plan == (("1",), ("2",), ("3",))


def test_temperature_intersection():
    # 1: [100,120] 2: [130,140] 3: [110,130]，容量任意
    # 1 与 2 不可同窑；1+3=[110,120] 可同窑(温度110)，2+3=[130,130](温度130)
    # 两种 2 窑方案：(1,3)(2) 与 (1)(2,3)；("1",) 是 ("1","3") 的前缀，
    # 字典序更小，故整体字典序最小方案为 [1] + [2,3]。
    items = [Ware("1", 1, 100, 120), Ware("2", 1, 130, 140), Ware("3", 1, 110, 130)]
    kilns = solve(items, 100)
    assert len(kilns) == 2
    assert kilns[0].item_ids == ("1",)
    assert kilns[0].firing_temp == 100
    assert kilns[1].item_ids == ("2", "3")
    assert kilns[1].firing_temp == 130


def test_firing_temp_is_intersection_lower_bound():
    items = [Ware("a", 1, 100, 300), Ware("b", 1, 150, 250), Ware("c", 1, 200, 400)]
    kilns = solve(items, 100)
    assert len(kilns) == 1
    assert kilns[0].firing_temp == 200  # max(100,150,200)
    assert kilns[0].total_volume == 3


def test_lex_tiebreak_example():
    # 构造窑次相同时的字典序选择：
    # 四件器物，每件体积 6，容量 10 -> 只能两两？不行，6+6>10 -> 每窑一件
    items = [Ware("b", 6, 1, 2), Ware("a", 6, 1, 2), Ware("c", 6, 1, 2)]
    k, plan = _run(items, 10)
    assert k == 3
    # 输入乱序，输出仍按编号升序、窑次按序列排序
    assert plan == (("a",), ("b",), ("c",))


def test_lex_tiebreak_pairing():
    # 容量 10，每件 5，温度全兼容 => 2 窑，两两组合。
    # 可能最优方案 (1,2)(3,4) / (1,3)(2,4) / (1,4)(2,3)，字典序最小为 (1,2)(3,4)
    items = [Ware(str(i), 5, 100, 200) for i in range(1, 5)]
    k, plan = _run(items, 10)
    assert k == 2
    assert plan == (("1", "2"), ("3", "4"))


def test_random_brute_force():
    rng = random.Random(42)
    for trial in range(300):
        n = rng.randint(1, 7)
        ids = [f"id{i}" for i in range(n)]
        capacity = rng.randint(5, 15)
        items = []
        for i in range(n):
            vol = rng.randint(1, 8)
            lo = rng.randint(900, 1000)
            hi = lo + rng.randint(0, 100)
            items.append(Ware(ids[i], vol, lo, hi))
        if any(it.volume > capacity for it in items):
            try:
                solve(items, capacity)
                assert False, f"trial {trial}: 应抛出 CapacityExceededError"
            except CapacityExceededError:
                continue
        got_k, got_plan = _run(items, capacity)
        exp_k, exp_plan = _brute_force(items, capacity)
        assert got_k == exp_k, f"trial {trial}: {got_k} != {exp_k}, {items}"
        assert got_plan == exp_plan, (
            f"trial {trial}: {got_plan} != {exp_plan}, items={items}, cap={capacity}"
        )


def test_n14_performance_and_correctness():
    # n=14 全兼容、容量只够单件 -> 14 窑，确保性能可接受
    items = [Ware(f"i{i:02d}", 5, 1000, 1000) for i in range(14)]
    import time

    t0 = time.time()
    k, plan = _run(items, 9)
    elapsed = time.time() - t0
    assert k == 14
    assert elapsed < 5.0, f"n=14 耗时 {elapsed:.2f}s"

    # 全部可同窑 -> 1 窑，烧成温度取交集下界
    items2 = [Ware(f"i{i:02d}", 1, 1000 + i, 2000) for i in range(14)]
    kilns = solve(items2, 1000)
    assert len(kilns) == 1
    assert kilns[0].firing_temp == 1013
    assert kilns[0].total_volume == 14


def test_min_capacity_basic():
    # 三件各 6、温度全兼容：3 窑需容量 6，2 窑需 12，1 窑需 18
    items = [Ware(str(i), 6, 100, 200) for i in range(1, 4)]
    cap, kilns = min_capacity_for_target(items, 3)
    assert cap == 6 and len(kilns) == 3
    cap, kilns = min_capacity_for_target(items, 2)
    assert cap == 12 and len(kilns) == 2
    # ("1",) 是 ("1","2") 的前缀，字典序更小，故首窑只装 "1"
    assert [k.item_ids for k in kilns] == [("1",), ("2", "3")]
    cap, kilns = min_capacity_for_target(items, 1)
    assert cap == 18 and len(kilns) == 1
    assert kilns[0].item_ids == ("1", "2", "3")
    assert kilns[0].firing_temp == 100
    assert kilns[0].total_volume == 18


def test_min_capacity_target_equals_item_count():
    # 目标 = 器物数时，最大单件体积即可（每件一窑）
    items = [Ware("a", 5, 100, 200), Ware("b", 9, 100, 200)]
    cap, kilns = min_capacity_for_target(items, 2)
    assert cap == 9 and len(kilns) == 2


def test_min_capacity_unreachable():
    # 温度区间互不兼容：任何容量下 1 与 2 都不可同窑
    items = [Ware("1", 3, 100, 120), Ware("2", 4, 130, 140)]
    try:
        min_capacity_for_target(items, 1)
        assert False, "应抛出 TargetUnreachableError"
    except TargetUnreachableError:
        pass
    # 目标 2 可达：最小容量为最大单件体积
    cap, kilns = min_capacity_for_target(items, 2)
    assert cap == 4 and len(kilns) == 2


def test_min_capacity_invalid_target():
    items = [Ware("a", 5, 100, 200)]
    try:
        min_capacity_for_target(items, 0)
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


def test_min_capacity_matches_linear_scan():
    # 随机小规模用例：与 [最大单件, 总体积] 全范围线性扫描交叉验证，
    # 同时校验返回方案与 solve 在该容量下的字典序最小方案一致
    rng = random.Random(7)
    for trial in range(200):
        n = rng.randint(1, 7)
        items = []
        for i in range(n):
            vol = rng.randint(1, 8)
            lo = rng.randint(900, 1000)
            hi = lo + rng.randint(0, 100)
            items.append(Ware(f"id{i}", vol, lo, hi))
        max_vol = max(it.volume for it in items)
        total = sum(it.volume for it in items)
        counts = {cap: len(solve(items, cap)) for cap in range(max_vol, total + 1)}
        for target in range(1, n + 1):
            feasible = [c for c, k in counts.items() if k <= target]
            if not feasible:
                try:
                    min_capacity_for_target(items, target)
                    assert False, f"trial {trial}: 应抛出 TargetUnreachableError"
                except TargetUnreachableError:
                    continue
            exp_cap = min(feasible)
            got_cap, kilns = min_capacity_for_target(items, target)
            assert got_cap == exp_cap, (
                f"trial {trial} target {target}: {got_cap} != {exp_cap}"
            )
            assert len(kilns) == counts[got_cap] <= target
            exp_kilns = solve(items, exp_cap)
            assert [k.item_ids for k in kilns] == [k.item_ids for k in exp_kilns]


def test_production_limit_count_then_volume():
    # 1 窑内最多 2 件：{p1,p3} 体积 9 与 {p2,p3} 体积 10 并列件数 -> 取体积大者
    items = [
        Ware("p1", 5, 100, 200),
        Ware("p2", 6, 120, 220),
        Ware("p3", 4, 150, 300),
        Ware("p4", 5, 1000, 1100),
    ]
    kept, deferred, kilns = select_within_kiln_limit(items, 10, 1)
    assert kept == ["p2", "p3"]
    assert deferred == ["p1", "p4"]
    assert len(kilns) == 1
    assert kilns[0].item_ids == ("p2", "p3")
    assert kilns[0].total_volume == 10
    assert kilns[0].firing_temp == 150


def test_production_limit_count_beats_volume():
    # 保留件数优先于总体积：两件小器物（10）胜过单件大器物（10 但仅 1 件）
    items = [Ware("big", 10, 1, 100), Ware("s1", 5, 1, 100), Ware("s2", 5, 1, 100)]
    kept, deferred, kilns = select_within_kiln_limit(items, 10, 1)
    assert kept == ["s1", "s2"]
    assert deferred == ["big"]
    assert len(kilns) == 1


def test_production_limit_lex_tiebreak():
    # 件数、体积均并列 -> 保留编号升序列表字典序最小
    items = [Ware("c", 5, 1, 100), Ware("a", 5, 1, 100), Ware("b", 5, 1, 100)]
    kept, deferred, kilns = select_within_kiln_limit(items, 10, 1)
    assert kept == ["a", "b"]
    assert deferred == ["c"]
    assert kilns[0].item_ids == ("a", "b")


def test_production_limit_temperature_select():
    # 1: [100,120] 2: [130,140] 3: [110,130]，限 1 窑：
    # 可行对 {1,3}（温度 110）与 {2,3}（温度 130）并列 -> 字典序取 {1,3}
    items = [Ware("1", 1, 100, 120), Ware("2", 1, 130, 140), Ware("3", 1, 110, 130)]
    kept, deferred, kilns = select_within_kiln_limit(items, 100, 1)
    assert kept == ["1", "3"]
    assert deferred == ["2"]
    assert kilns[0].firing_temp == 110


def test_production_limit_keeps_all_when_limit_allows():
    # 限制不小于最少窑次时保留全部，装窑与 solve 一致
    items = [Ware(str(i), 6, 100, 200) for i in range(1, 4)]
    kept, deferred, kilns = select_within_kiln_limit(items, 10, 3)
    assert kept == ["1", "2", "3"]
    assert deferred == []
    assert [k.item_ids for k in kilns] == [("1",), ("2",), ("3",)]


def test_production_limit_invalid_max_kilns():
    items = [Ware("a", 5, 100, 200)]
    try:
        select_within_kiln_limit(items, 10, 0)
        assert False, "应抛出 ValueError"
    except ValueError:
        pass


def test_production_limit_n14_performance():
    # n=14、每窑一件：限 7 窑 -> 保留字典序最小的 7 件，耗时应可接受
    import time

    items = [Ware(f"i{i:02d}", 5, 1000, 1000) for i in range(14)]
    t0 = time.time()
    kept, deferred, kilns = select_within_kiln_limit(items, 9, 7)
    elapsed = time.time() - t0
    assert kept == [f"i{i:02d}" for i in range(7)]
    assert deferred == [f"i{i:02d}" for i in range(7, 14)]
    assert len(kilns) == 7
    assert elapsed < 5.0, f"n=14 耗时 {elapsed:.2f}s"


def _brute_force_production_limit(items, capacity, max_kilns):
    """枚举全部子集：用 solve 求子集最少窑次，按规则选最优保留集合。"""
    best = None  # (-件数, -体积, 编号升序元组)，取最小者
    n = len(items)
    for bits in range(1 << n):
        subset = [items[i] for i in range(n) if (bits >> i) & 1]
        if not subset:
            continue
        if len(solve(subset, capacity)) > max_kilns:
            continue
        ids = tuple(sorted(it.item_id for it in subset))
        vol = sum(it.volume for it in subset)
        key = (-len(ids), -vol, ids)
        if best is None or key < best:
            best = key
    return best


def test_production_limit_random_cross_check():
    rng = random.Random(2026)
    for trial in range(200):
        n = rng.randint(1, 8)
        capacity = rng.randint(5, 15)
        items = []
        for i in range(n):
            vol = rng.randint(1, capacity)  # 保证单件不超容量
            lo = rng.randint(900, 1000)
            hi = lo + rng.randint(0, 100)
            items.append(Ware(f"id{i}", vol, lo, hi))
        max_kilns = rng.randint(1, n)

        kept, deferred, kilns = select_within_kiln_limit(items, capacity, max_kilns)

        exp = _brute_force_production_limit(items, capacity, max_kilns)
        assert exp is not None, f"trial {trial}: 至少单件可选，不应为空"
        exp_count, exp_vol, exp_ids = -exp[0], -exp[1], exp[2]
        assert tuple(kept) == exp_ids, f"trial {trial}: {kept} != {list(exp_ids)}"
        assert len(kept) == exp_count
        assert sum(k.total_volume for k in kilns) == exp_vol

        # 延期 = 全集 - 保留，双方均按编号升序
        all_ids = sorted(it.item_id for it in items)
        assert tuple(deferred) == tuple(i for i in all_ids if i not in set(exp_ids))

        # 装窑与对保留集合直接 solve 一致（最少窑次 + 字典序），且不超限制
        kept_items = [it for it in items if it.item_id in set(kept)]
        exp_kilns = solve(kept_items, capacity)
        assert [k.item_ids for k in kilns] == [k.item_ids for k in exp_kilns]
        assert len(kilns) <= max_kilns


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n全部 {len(fns)} 个测试通过")
