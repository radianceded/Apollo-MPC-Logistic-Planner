"""
MPC Controller v3 for Apollo 9.0
Changes from v2:
- Outputs ADCTrajectory to /apollo/planning (instead of ControlCommand to /apollo/control)
- MPC solve -> bicycle model forward simulation -> pack as ADCTrajectory
- Sim_Control executes the trajectory directly
- Reads reference trajectory from /apollo/planning (need to distinguish own vs Planning's)
- Run-as-fast-as-possible loop (no fixed frequency cap)
"""

import sys
import math
import time
import threading
import numpy as np

sys.path.insert(0, '/opt/apollo/neo/bin/plot_control.runfiles/apollo_src/cyber/python')
sys.path.insert(0, '/opt/apollo/neo/bin/plot_control.runfiles/apollo_src/cyber/python/internal')
sys.path.insert(0, '/opt/apollo/neo/bin/plot_control.runfiles/apollo_src')
sys.path.insert(0, '/opt/apollo/neo/python')

from cyber.python.cyber_py3 import cyber
from modules.common_msgs.planning_msgs import planning_pb2
from modules.common_msgs.localization_msgs import localization_pb2

import cvxpy

# ============================================================
# MPC Parameters
# ============================================================
NX = 4  # state: [x, y, v, yaw]
NU = 2  # control: [acceleration, steering_angle]
T = 5   # prediction horizon steps
DT = 0.1  # time step [s]

# Cost matrices
Q = np.diag([1.0, 1.0, 0.3, 3.0])   # state tracking cost
Qf = np.diag([1.0, 1.0, 0.3, 3.0])  # terminal cost
R = np.diag([0.1, 0.1])              # control cost
Rd = np.diag([0.1, 0.5])             # control change cost

# Forward simulation horizon (longer than MPC horizon for smooth trajectory)
SIM_STEPS = 50  # 50 * 0.1s = 5.0s lookahead

# ============================================================
# Vehicle Parameters (Lincoln MKZ)
# ============================================================
WB = 2.85          # wheelbase [m]
MAX_STEER = np.deg2rad(35.0)
MAX_DSTEER = np.deg2rad(30.0)
MAX_SPEED = 15.0
MIN_SPEED = -2.0
MAX_ACCEL = 2.0
MAX_DECEL = -3.0

# Module name to tag our own messages
MPC_MODULE_NAME = "mpc_controller"


# ============================================================
# Utility
# ============================================================
def angle_mod(x):
    """Normalize angle to [-pi, pi]"""
    return (x + math.pi) % (2 * math.pi) - math.pi


def get_linear_model_matrix(v, phi, delta):
    """Linearized bicycle kinematic model."""
    v_lin = max(abs(v), 0.5)
    delta = max(-MAX_STEER * 0.9, min(MAX_STEER * 0.9, delta))

    A = np.eye(NX)
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = -DT * v_lin * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v_lin * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v_lin / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v_lin * math.sin(phi) * phi
    C[1] = -DT * v_lin * math.cos(phi) * phi
    C[3] = -DT * v_lin * delta / (WB * math.cos(delta) ** 2)

    return A, B, C


def solve_mpc(xref, x0, dref):
    """Single-shot linearized MPC with OSQP."""
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)
        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(
            xref[2, t], xref[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [u[0, :] <= MAX_ACCEL]
    constraints += [u[0, :] >= MAX_DECEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True,
               eps_abs=1e-3, eps_rel=1e-3, max_iter=200)

    status = prob.status
    if status == cvxpy.OPTIMAL or status == cvxpy.OPTIMAL_INACCURATE:
        oa = np.array(u.value[0, :]).flatten()
        od = np.array(u.value[1, :]).flatten()
        return oa, od, status
    else:
        return None, None, status


# ============================================================
# Forward simulation (bicycle model)
# ============================================================
def forward_simulate(x0, oa, od, n_steps):
    """
    Forward simulate bicycle model using MPC control sequence.
    For steps beyond MPC horizon T, hold last control input constant.
    
    Args:
        x0: initial state [x, y, v, yaw]
        oa: acceleration sequence (length T)
        od: steering sequence (length T)
        n_steps: total simulation steps
    
    Returns:
        states: array of shape (n_steps+1, 4) - [x, y, v, yaw] at each step
        accels: array of shape (n_steps,) - acceleration at each step
        steers: array of shape (n_steps,) - steering at each step
    """
    states = np.zeros((n_steps + 1, NX))
    states[0] = x0
    accels = np.zeros(n_steps)
    steers = np.zeros(n_steps)

    for i in range(n_steps):
        # Use MPC control if available, otherwise hold last value
        if i < len(oa):
            a = oa[i]
            delta = od[i]
        else:
            a = oa[-1]
            delta = od[-1]

        accels[i] = a
        steers[i] = delta

        x, y, v, yaw = states[i]
        # Bicycle kinematic model
        x_next = x + v * math.cos(yaw) * DT
        y_next = y + v * math.sin(yaw) * DT
        v_next = v + a * DT
        v_next = max(MIN_SPEED, min(MAX_SPEED, v_next))
        yaw_next = yaw + v * math.tan(delta) / WB * DT

        states[i + 1] = [x_next, y_next, v_next, yaw_next]

    return states, accels, steers


# ============================================================
# ADCTrajectory packing
# ============================================================
def pack_trajectory(states, accels, steers, timestamp):
    """
    Pack forward-simulated states into ADCTrajectory protobuf.
    
    Args:
        states: (n_steps+1, 4) array of [x, y, v, yaw]
        accels: (n_steps,) acceleration at each step
        steers: (n_steps,) steering angle at each step
        timestamp: current time in seconds
    
    Returns:
        ADCTrajectory protobuf message
    """
    traj = planning_pb2.ADCTrajectory()

    # Header
    traj.header.timestamp_sec = timestamp
    traj.header.module_name = MPC_MODULE_NAME
    traj.header.sequence_num = int(timestamp * 1000) % 2000000000

    # Required fields
    traj.gear = 1  # GEAR_DRIVE
    traj.engage_advice.advice = 1  # READY_TO_ENGAGE
    traj.estop.is_estop = False

    # Compute cumulative arc length
    n_points = len(states)
    s_cumul = np.zeros(n_points)
    for i in range(1, n_points):
        dx = states[i, 0] - states[i - 1, 0]
        dy = states[i, 1] - states[i - 1, 1]
        s_cumul[i] = s_cumul[i - 1] + math.sqrt(dx * dx + dy * dy)

    # Pack trajectory points
    for i in range(n_points):
        pt = traj.trajectory_point.add()
        pt.path_point.x = states[i, 0]
        pt.path_point.y = states[i, 1]
        pt.path_point.theta = states[i, 3]
        pt.path_point.s = s_cumul[i]
        pt.v = states[i, 2]
        pt.relative_time = i * DT

        # Curvature from steering angle: kappa = tan(delta) / WB
        if i < len(steers):
            pt.path_point.kappa = math.tan(steers[i]) / WB
        else:
            pt.path_point.kappa = math.tan(steers[-1]) / WB

        # Acceleration
        if i < len(accels):
            pt.a = accels[i]
        else:
            pt.a = 0.0

    return traj


def pack_fallback_trajectory(ego_x, ego_y, ego_v, ego_yaw, steer, accel, timestamp):
    """
    Pack a simple trajectory from PID fallback (straight-line forward sim).
    """
    oa = np.full(SIM_STEPS, accel)
    od = np.full(SIM_STEPS, steer)
    x0 = np.array([ego_x, ego_y, ego_v, ego_yaw])
    states, accels, steers = forward_simulate(x0, oa, od, SIM_STEPS)
    return pack_trajectory(states, accels, steers, timestamp)


# ============================================================
# Trajectory matching
# ============================================================
def calc_nearest_index(x, y, traj_points):
    """Find nearest trajectory point."""
    n = len(traj_points)
    min_dist = float('inf')
    min_idx = 0

    for i in range(n):
        p = traj_points[i]
        dx = x - p.path_point.x
        dy = y - p.path_point.y
        dist = dx * dx + dy * dy
        if dist < min_dist:
            min_dist = dist
            min_idx = i

    return min_idx, math.sqrt(min_dist)


def calc_ref_trajectory(ego_x, ego_y, ego_v, ego_yaw, traj_points):
    """Build reference trajectory with proper angle normalization."""
    n = len(traj_points)
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T))

    nearest_idx, _ = calc_nearest_index(ego_x, ego_y, traj_points)

    if n >= 2:
        p0 = traj_points[nearest_idx].path_point
        p1 = traj_points[min(nearest_idx + 1, n - 1)].path_point
        dl = math.sqrt((p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2)
        if dl < 0.01:
            dl = 0.1
    else:
        dl = 0.1

    speed_for_step = max(abs(ego_v), 1.0)
    idx_step = max(1, int(speed_for_step * DT / dl))

    for i in range(T + 1):
        idx = min(nearest_idx + i * idx_step, n - 1)
        p = traj_points[idx]
        xref[0, i] = p.path_point.x
        xref[1, i] = p.path_point.y
        xref[2, i] = p.v
        raw_theta = p.path_point.theta
        if i == 0:
            xref[3, i] = ego_yaw + angle_mod(raw_theta - ego_yaw)
        else:
            xref[3, i] = xref[3, i - 1] + angle_mod(raw_theta - xref[3, i - 1])

    for i in range(T):
        idx = min(nearest_idx + i * idx_step, n - 1)
        p = traj_points[idx]
        kappa = p.path_point.kappa
        dref[0, i] = math.atan(kappa * WB)

    return xref, nearest_idx, dref


def pid_fallback(ego_x, ego_y, ego_v, ego_yaw, traj_points):
    """Simple PID fallback when MPC fails."""
    nearest_idx, dist = calc_nearest_index(ego_x, ego_y, traj_points)
    n = len(traj_points)

    look_idx = min(nearest_idx + 5, n - 1)
    target = traj_points[look_idx]

    dx = target.path_point.x - ego_x
    dy = target.path_point.y - ego_y
    target_heading = math.atan2(dy, dx)
    heading_err = angle_mod(target_heading - ego_yaw)

    steer = 1.5 * heading_err
    steer = max(-MAX_STEER, min(MAX_STEER, steer))

    target_v = target.v
    accel = 1.0 * (target_v - ego_v)
    accel = max(MAX_DECEL, min(MAX_ACCEL, accel))

    return steer, accel


# ============================================================
# Main Controller
# ============================================================
class MPCController:
    def __init__(self):
        self.lock = threading.Lock()
        self.ref_traj_points = None  # reference from Planning module
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_yaw = 0.0
        self.ego_v = 0.0
        self.localization_received = False
        self.planning_received = False
        self.seq = 0

        cyber.init()
        self.node = cyber.Node("mpc_controller_node")

        # Writer: publish ADCTrajectory to /apollo/planning
        self.traj_writer = self.node.create_writer(
            "/apollo/planning", planning_pb2.ADCTrajectory)

        # Reader: subscribe to /apollo/planning to get reference trajectory
        # We filter out our own messages by module_name
        self.node.create_reader(
            "/apollo/planning", planning_pb2.ADCTrajectory,
            self.planning_callback)

        # Reader: localization
        self.node.create_reader(
            "/apollo/localization/pose", localization_pb2.LocalizationEstimate,
            self.localization_callback)

        print("[MPC] Controller v3 initialized (ADCTrajectory output)")
        print(f"[MPC] T={T}, DT={DT}, SIM_STEPS={SIM_STEPS}, WB={WB}")
        print(f"[MPC] solver=OSQP, module_name={MPC_MODULE_NAME}")

    def planning_callback(self, msg):
        """
        Receive planning messages. Only store reference if it's NOT our own.
        Our own messages have module_name == MPC_MODULE_NAME.
        """
        if msg.header.module_name == MPC_MODULE_NAME:
            return  # skip our own output

        with self.lock:
            pts = list(msg.trajectory_point)
            if len(pts) > 0:
                self.ref_traj_points = pts
                self.planning_received = True

    def localization_callback(self, msg):
        with self.lock:
            pose = msg.pose
            self.ego_x = pose.position.x
            self.ego_y = pose.position.y
            self.ego_yaw = pose.heading
            vx = pose.linear_velocity.x
            vy = pose.linear_velocity.y
            self.ego_v = math.sqrt(vx * vx + vy * vy)
            self.localization_received = True

    def run(self):
        print("[MPC] Waiting for data...")

        while not cyber.is_shutdown():
            time.sleep(0.05)
            if self.planning_received and self.localization_received:
                print("[MPC] Data received, starting control loop")
                break

        loop_count = 0
        mpc_ok_count = 0
        pid_count = 0

        while not cyber.is_shutdown():
            t_start = time.time()

            with self.lock:
                traj_points = self.ref_traj_points
                ego_x = self.ego_x
                ego_y = self.ego_y
                ego_yaw = self.ego_yaw
                ego_v = self.ego_v

            if traj_points is None or len(traj_points) < T + 1:
                time.sleep(0.05)
                continue

            # Build reference from Planning's trajectory
            xref, nearest_idx, dref = calc_ref_trajectory(
                ego_x, ego_y, ego_v, ego_yaw, traj_points)

            x0 = np.array([ego_x, ego_y, ego_v, xref[3, 0]])

            use_pid = False
            timestamp = time.time()

            try:
                oa, od, status = solve_mpc(xref, x0, dref)
                if oa is not None:
                    # Forward simulate full trajectory
                    states, accels, steers = forward_simulate(
                        x0, oa, od, SIM_STEPS)
                    traj_msg = pack_trajectory(
                        states, accels, steers, timestamp)
                    mpc_ok_count += 1
                else:
                    # PID fallback -> also forward simulate
                    steer, accel = pid_fallback(
                        ego_x, ego_y, ego_v, ego_yaw, traj_points)
                    traj_msg = pack_fallback_trajectory(
                        ego_x, ego_y, ego_v, ego_yaw,
                        steer, accel, timestamp)
                    use_pid = True
                    pid_count += 1
            except Exception as e:
                steer, accel = pid_fallback(
                    ego_x, ego_y, ego_v, ego_yaw, traj_points)
                traj_msg = pack_fallback_trajectory(
                    ego_x, ego_y, ego_v, ego_yaw,
                    steer, accel, timestamp)
                use_pid = True
                pid_count += 1
                status = f"err:{e}"

            # Publish
            self.traj_writer.write(traj_msg)

            # Logging
            loop_count += 1
            t_elapsed = time.time() - t_start
            if loop_count % 10 == 0:
                nearest_p = traj_points[min(nearest_idx, len(traj_points) - 1)]
                dx = ego_x - nearest_p.path_point.x
                dy = ego_y - nearest_p.path_point.y
                lat_err = math.sqrt(dx * dx + dy * dy)
                ref_theta = nearest_p.path_point.theta
                yaw_err = math.degrees(angle_mod(ego_yaw - ref_theta))
                mode = "PID" if use_pid else "MPC"
                n_pts = len(traj_msg.trajectory_point)
                print(f"[{mode}] #{loop_count} | "
                      f"pos=({ego_x:.1f}, {ego_y:.1f}) | "
                      f"v={ego_v:.2f} | "
                      f"lat={lat_err:.2f}m | "
                      f"yaw_e={yaw_err:.1f}d | "
                      f"{t_elapsed*1000:.0f}ms | "
                      f"pts={n_pts} | "
                      f"ok={mpc_ok_count} pid={pid_count}")

            # No fixed sleep - run as fast as possible
            # (solver takes ~170ms avg, that's the natural rate)


def main():
    print("=" * 60)
    print("  Apollo MPC Controller v3 (ADCTrajectory output)")
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    controller = MPCController()
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n[MPC] Shutting down...")


if __name__ == "__main__":
    main()
