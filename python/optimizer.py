import sys, json
from ortools.sat.python import cp_model

def _expand_prices(hour_prices, slots_per_hour, horizon_hours, slot_hours):
    if not hour_prices:
        hour_prices = [0.0]*24
    prices=[]
    for h in range(horizon_hours):
        v=float(hour_prices[h % 24])*slot_hours
        for _ in range(slots_per_hour):
            prices.append(v)
    return prices

def optimize(hour_prices, jobs, cfg):
    slot_min=int(cfg.get("time_slot_minutes",15))
    horizon_h=int(cfg.get("horizon_hours",24))
    capacity_kw=float(cfg.get("capacity_kw",150.0))
    demand_per_kw=float(cfg.get("demand_charge_per_kw",0.0))

    slots_per_hour=max(1, 60//slot_min)
    T=horizon_h*slots_per_hour
    slot_hours=slot_min/60.0
    prices_slots=_expand_prices(hour_prices, slots_per_hour, horizon_h, slot_hours)

    KW_SCALE=10
    COST_SCALE=1000
    cap=int(round(capacity_kw*KW_SCALE))

    m=cp_model.CpModel()
    J=len(jobs)

    # one-hot start time
    x=[[None]*T for _ in range(J)]
    for j, job in enumerate(jobs):
        dur=int(job["duration_slots"])
        earliest=int(job.get("earliest_slot",0))
        latest_finish=int(job.get("latest_finish_slot",T))
        feas=[]
        for t in range(0,T-dur+1):
            if t>=earliest and t+dur<=latest_finish:
                x[j][t]=m.NewBoolVar(f"x_{j}_{t}")
                feas.append(x[j][t])
        if feas:
            m.Add(sum(feas)==1)
        else:
            m.AddBoolOr([])  # infeasible guard

    # run[j][t] whether job j runs at slot t
    run=[[m.NewBoolVar(f"run_{j}_{t}") for t in range(T)] for j in range(J)]
    for j, job in enumerate(jobs):
        dur=int(job["duration_slots"])
        for t in range(T):
            covers=[]
            for s in range(max(0,t-dur+1), min(t, T-dur)+1):
                if x[j][s] is not None:
                    covers.append(x[j][s])
            m.Add(run[j][t] == (sum(covers) if covers else 0))

    # capacity and peak
    load=[m.NewIntVar(0, cap, f"load_{t}") for t in range(T)]
    power_int=[int(round(float(jobs[j]["power_kw"])*KW_SCALE)) for j in range(J)]
    for t in range(T):
        m.Add(load[t]==sum(power_int[j]*run[j][t] for j in range(J)))
    peak=m.NewIntVar(0, cap, "peak")
    for t in range(T):
        m.Add(load[t] <= peak)

    # cost: energy + demand
    price_int=[int(round(p*COST_SCALE)) for p in prices_slots]
    energy_cost=sum(price_int[t]*load[t] for t in range(T))
    demand_cost=int(round(demand_per_kw*COST_SCALE/KW_SCALE))*peak
    m.Minimize(energy_cost + demand_cost)

    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=30.0
    status=solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"ok": False, "error":"no feasible schedule"}

    # outputs
    starts=[]
    for j, job in enumerate(jobs):
        dur=int(job["duration_slots"])
        start_t=None
        for t in range(T-dur+1):
            if x[j][t] is not None and solver.Value(x[j][t])==1:
                start_t=t
                break
        starts.append({
            "job_id": job["job_id"],
            "start_slot": start_t,
            "end_slot": (start_t+dur) if start_t is not None else None,
            "power_kw": float(job["power_kw"]),
        })

    load_kw=[solver.Value(load[t])/KW_SCALE for t in range(T)]
    peak_kw=max(load_kw) if load_kw else 0.0
    energy_cost_real=sum(prices_slots[t]*load_kw[t] for t in range(T))
    demand_cost_real=demand_per_kw*peak_kw
    total=energy_cost_real + demand_cost_real

    return {
        "ok": True,
        "energy_cost": energy_cost_real,
        "demand_cost": demand_cost_real,
        "peak_kw": peak_kw,
        "starts": starts,
        "per_slot_load": load_kw,
        "slot_hours": slot_hours,
        "total_cost": total
    }

if __name__ == "__main__":
    payload=json.loads(sys.stdin.read() or "{}")
    body=optimize(
        payload.get("hour_prices",[]),
        payload.get("jobs",[]),
        payload.get("config",{})
    )
    print(json.dumps(body))