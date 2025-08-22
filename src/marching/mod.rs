use nalgebra::{Matrix2, Point2, Vector2, Vector3};

use crate::{
    misc::{FloatingPoint, Plane},
    surface::NurbsSurface3D,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Active {
    None,
    UMin,
    UMax,
    VMin,
    VMax,
    Corner,
}

/// Predictor variants
#[derive(Debug, Clone, Copy)]
pub enum Predictor {
    Euler,
    RK2,
}

#[derive(Debug, Clone, Copy)]
pub struct MarchConfig<T: FloatingPoint> {
    /// Target 3D arc length step (initial value)
    pub step_arc_length: T,
    /// Minimum step size (stop if reduced below this)
    pub min_step_arc_length: T,
    /// Tolerance for Newton / convergence (g=0, and auxiliary equation)
    pub tolerance: T,
    /// Maximum number of steps per direction
    pub max_steps: usize,
    /// UV distance threshold for closed loop detection
    pub close_tol_uv: T,
    /// Minimum number of samples required for closed loop detection
    pub close_min_count: usize,
    /// Type of predictor
    pub predictor: Predictor,
}

impl<T: FloatingPoint> Default for MarchConfig<T> {
    fn default() -> Self {
        Self {
            step_arc_length: T::from_f64(0.1).unwrap(),
            min_step_arc_length: T::from_f64(1e-6).unwrap(),
            tolerance: T::from_f64(1e-9).unwrap(),
            max_steps: 10_000,
            close_tol_uv: T::from_f64(1e-5).unwrap(),
            close_min_count: 16,
            predictor: Predictor::Euler,
        }
    }
}

#[derive(Debug)]
pub struct MarchResult<T: FloatingPoint> {
    pub uv: Vec<Point2<T>>,
    pub closed: bool,
    pub hit_boundary: bool,
}

#[allow(dead_code)]
fn g<T: FloatingPoint>(s: &NurbsSurface3D<T>, pl: &Plane<T>, u: T, v: T) -> T {
    pl.signed_distance(&s.point_at(u, v))
}

/// Get the 3D tangent and UV velocity of the intersection curve.
/// Returns: (t_uv, |T3|, su, sv)  — t_uv=(g_v, -g_u), T3 = (n·S_v)S_u - (n·S_u)S_v
fn uv_velocity<T: FloatingPoint>(
    s: &NurbsSurface3D<T>,
    pl: &Plane<T>,
    u: T,
    v: T,
) -> Option<(Vector2<T>, T, Vector3<T>, Vector3<T>)> {
    let deriv = s.rational_derivatives(u, v, 1);
    let su = deriv[1][0].clone();
    let sv = deriv[0][1].clone();
    let gu = pl.normal().dot(&su);
    let gv = pl.normal().dot(&sv);
    let t_uv = Vector2::new(gv, -gu);
    // 3D 接線（スケールは t_uv に一致）
    let t3 = su.clone() * gv - sv.clone() * gu;
    let len3 = t3.norm();
    if len3 < T::from_f64(1e-18).unwrap() {
        None
    } else {
        Some((t_uv, len3, su, sv))
    }
}

/// Get the maximum alpha and the boundary hit when moving in the direction of p=(du,dv) within 0<alpha<=1.
fn max_alpha_to_box<T: FloatingPoint>(
    u: T,
    v: T,
    p: Vector2<T>,
    umin: T,
    umax: T,
    vmin: T,
    vmax: T,
) -> (T, Active) {
    let mut alpha = T::one();
    let mut act = Active::None;
    let eps = T::from_f64(1e-15).unwrap();

    // u
    if p.x > eps {
        let a = (umax - u) / p.x;
        if a < alpha {
            alpha = a;
            act = Active::UMax;
        }
    } else if p.x < -eps {
        let a = (umin - u) / p.x; // 負/負 → 正
        if a < alpha {
            alpha = a;
            act = Active::UMin;
        }
    }
    // v
    if p.y > eps {
        let a = (vmax - v) / p.y;
        if (a - alpha).abs() <= T::from_f64(1e-12).unwrap() && act != Active::None {
            act = Active::Corner;
        } else if a < alpha {
            alpha = a;
            act = Active::VMax;
        }
    } else if p.y < -eps {
        let a = (vmin - v) / p.y;
        if (a - alpha).abs() <= T::from_f64(1e-12).unwrap() && act != Active::None {
            act = Active::Corner;
        } else if a < alpha {
            alpha = a;
            act = Active::VMin;
        }
    }

    if alpha < T::zero() {
        alpha = T::zero();
    }
    if alpha > T::one() {
        alpha = T::one();
    }
    (alpha, act)
}

/// Predictor with box constraints (interior mode).
fn predictor_interior<T: FloatingPoint>(
    s: &NurbsSurface3D<T>,
    pl: &Plane<T>,
    u: T,
    v: T,
    step_3d: T,
    mode: Predictor,
    sign: T,
) -> Option<(T, T, Vector2<T>, Active)> {
    let ((umin, umax), (vmin, vmax)) = s.knots_domain();

    match mode {
        Predictor::Euler => {
            let (t_uv, len3, _, _) = uv_velocity(s, pl, u, v)?;
            let t_uv = t_uv * sign;
            let h = step_3d / len3.max(T::from_f64(1e-18).unwrap());
            let p = t_uv * h;

            let (alpha, hit) = max_alpha_to_box(u, v, p, umin, umax, vmin, vmax);
            let mut t_uvn = t_uv;
            if t_uvn.norm() > T::zero() {
                t_uvn = t_uvn.normalize();
            }
            Some((u + alpha * p.x, v + alpha * p.y, t_uvn, hit))
        }
        Predictor::RK2 => {
            // 中点法（箱制約は最後に適用）
            let (t_uv1, len31, _, _) = uv_velocity(s, pl, u, v)?;
            let k1 = t_uv1 * sign / len31.max(T::from_f64(1e-18).unwrap());
            let um = u + T::from_f64(0.5).unwrap() * step_3d * k1.x;
            let vm = v + T::from_f64(0.5).unwrap() * step_3d * k1.y;
            let (t_uv2, len32, _, _) = uv_velocity(s, pl, um, vm)?;
            let k2 = t_uv2 * sign / len32.max(T::from_f64(1e-18).unwrap());
            let p = k2 * step_3d;

            let (alpha, hit) = max_alpha_to_box(u, v, p, umin, umax, vmin, vmax);
            let mut dir = t_uv2 * sign;
            if dir.norm() > T::zero() {
                dir = dir.normalize();
            }
            Some((u + alpha * p.x, v + alpha * p.y, dir, hit))
        }
    }
}

/// Predictor for boundary mode (move along u=const or v=const).
fn predictor_boundary<T: FloatingPoint>(
    s: &NurbsSurface3D<T>,
    pl: &Plane<T>,
    u: T,
    v: T,
    step_3d: T,
    active: Active,
    sign: T,
) -> Option<(T, T, Vector2<T>, Active)> {
    let ((umin, umax), (vmin, vmax)) = s.knots_domain();
    let (_, _, su, sv) = uv_velocity(s, pl, u, v)?; // su, sv を利用

    // Move only in the free direction (adjust the 3D step size).
    let p = match active {
        Active::UMin | Active::UMax => {
            let lv = sv.norm().max(T::from_f64(1e-18).unwrap());
            Vector2::new(T::zero(), sign * step_3d / lv)
        }
        Active::VMin | Active::VMax => {
            let lu = su.norm().max(T::from_f64(1e-18).unwrap());
            Vector2::new(sign * step_3d / lu, T::zero())
        }
        _ => return None,
    };

    let (alpha, hit2) = max_alpha_to_box(u, v, p, umin, umax, vmin, vmax);
    let hit = match (active, hit2) {
        // If already at UMin and hit VMax, then Corner
        (Active::UMin, Active::VMax)
        | (Active::UMin, Active::VMin)
        | (Active::UMax, Active::VMax)
        | (Active::UMax, Active::VMin)
        | (Active::VMin, Active::UMax)
        | (Active::VMin, Active::UMin)
        | (Active::VMax, Active::UMax)
        | (Active::VMax, Active::UMin) => Active::Corner,
        // 同一境界に沿った移動なら active 継続
        _ => active,
    };

    // "Apparent tangent direction" (for display purposes, to match internal conventions)
    let dir = match active {
        Active::UMin | Active::UMax => Vector2::new(T::zero(), sign).normalize(),
        Active::VMin | Active::VMax => Vector2::new(sign, T::zero()).normalize(),
        _ => Vector2::new(T::zero(), T::zero()),
    };

    Some((u + alpha * p.x, v + alpha * p.y, dir, hit))
}

/// 2×2 Newton corrector:
/// active=None → { g(u,v)=0,  (Δu,Δv)・t_uv = 0 } (back to the shortest path)
/// active=UMin/UMax → { g(u,v)=0, u - ub = 0 }
/// active=VMin/VMax → { g(u,v)=0, v - vb = 0 }
/// Returns: (u, v, active)
fn corrector_constrained<T: FloatingPoint>(
    s: &NurbsSurface3D<T>,
    pl: &Plane<T>,
    mut u: T,
    mut v: T,
    uv_tan: Vector2<T>,
    active: Active,
    tol: T,
) -> Option<(T, T, Active)> {
    let ((umin, umax), (vmin, vmax)) = s.knots_domain();
    let (ub, vb) = match active {
        Active::UMin => (Some(umin), None),
        Active::UMax => (Some(umax), None),
        Active::VMin => (None, Some(vmin)),
        Active::VMax => (None, Some(vmax)),
        _ => (None, None),
    };

    for _ in 0..30 {
        let deriv = s.rational_derivatives(u, v, 1);
        let su = &deriv[1][0];
        let sv = &deriv[0][1];
        let gu = pl.normal().dot(&su);
        let gv = pl.normal().dot(&sv);
        let f1 = pl.signed_distance(&s.point_at(u, v)); // g(u,v)

        let (f2, j20, j21) = match active {
            Active::UMin | Active::UMax => ((u - ub.unwrap()), T::one(), T::zero()),
            Active::VMin | Active::VMax => ((v - vb.unwrap()), T::zero(), T::one()),
            Active::None | Active::Corner => (T::zero(), uv_tan.x, uv_tan.y), // 方向制約
        };

        let j = Matrix2::new(gu, gv, j20, j21);
        let rhs = nalgebra::Vector2::new(-f1, -f2);

        if let Some(delta) = j.lu().solve(&rhs) {
            u += delta.x;
            v += delta.y;
            if f1.abs() < tol && f2.abs() < tol && delta.amax() < tol {
                u = u.clamp(umin, umax);
                v = v.clamp(vmin, vmax);
                // If close to the boundary, keep the active state. If far away, return to None.
                let new_active = if let (Some(ub), _) = (ub, vb) {
                    if (u - ub).abs() <= T::from_f64(10.0).unwrap() * tol {
                        active
                    } else {
                        Active::None
                    }
                } else if let (_, Some(vb)) = (ub, vb) {
                    if (v - vb).abs() <= T::from_f64(10.0).unwrap() * tol {
                        active
                    } else {
                        Active::None
                    }
                } else {
                    Active::None
                };
                return Some((u, v, new_active));
            }
        } else {
            return None;
        }
    }
    None
}

/// March in one direction with box constraints and boundary mode.
fn march_one_direction<T: FloatingPoint>(
    s: &NurbsSurface3D<T>,
    pl: &Plane<T>,
    uv0: (T, T),
    start_active: Active,
    cfg: MarchConfig<T>,
    sign: T,
) -> (Vec<Point2<T>>, bool, bool) {
    let mut pts = Vec::<Point2<T>>::with_capacity(cfg.max_steps + 1);
    let mut u = uv0.0;
    let mut v = uv0.1;
    let mut active = start_active;
    let ((umin, umax), (vmin, vmax)) = s.knots_domain();

    pts.push(Point2::new(u, v));

    let mut step = cfg.step_arc_length;
    let mut closed = false;
    let mut hit_boundary_any = matches!(
        active,
        Active::UMin | Active::UMax | Active::VMin | Active::VMax
    );

    for i in 0..cfg.max_steps {
        // Predictor
        let pred = if matches!(active, Active::None) {
            predictor_interior(s, pl, u, v, step, cfg.predictor, sign)
        } else if matches!(active, Active::Corner) {
            None
        } else {
            predictor_boundary(s, pl, u, v, step, active, sign)
        };

        let (up, vp, tan_hat, hit) = match pred {
            Some(x) => x,
            None => break,
        };

        // Corrector
        match corrector_constrained(s, pl, up, vp, tan_hat, hit, cfg.tolerance) {
            Some((uc, vc, act_new)) => {
                // 足踏み／ゼロ移動を検出
                let du = uc - u;
                let dv = vc - v;
                if du.hypot(dv) < T::from_f64(1e-12).unwrap() {
                    // Step down and try again
                    step = step * T::from_f64(0.5).unwrap();
                    if step < cfg.min_step_arc_length {
                        break;
                    }
                    continue;
                }
                u = uc;
                v = vc;
                active = act_new;
                if matches!(
                    active,
                    Active::UMin | Active::UMax | Active::VMin | Active::VMax
                ) {
                    hit_boundary_any = true;
                }
                pts.push(Point2::new(u, v));

                // Check for closed loop (close to start point after a sufficient distance)
                if i > cfg.close_min_count {
                    let d0 = (u - uv0.0).hypot(v - uv0.1);
                    if d0 <= cfg.close_tol_uv {
                        closed = true;
                        break;
                    }
                }

                // If successful, step back a little (modest upward adjustment)
                step = (step * T::from_f64(1.2).unwrap()).min(cfg.step_arc_length);
            }
            None => {
                // 収束失敗 → バックトラック
                step = step * T::from_f64(0.5).unwrap();
                if step < cfg.min_step_arc_length {
                    break;
                }
                continue;
            }
        }

        // If outside the domain, stop (safety measure)
        if !(umin - T::from_f64(1e-12).unwrap()..=umax + T::from_f64(1e-12).unwrap()).contains(&u)
            || !(vmin - T::from_f64(1e-12).unwrap()..=vmax + T::from_f64(1e-12).unwrap())
                .contains(&v)
        {
            break;
        }
    }

    (pts, closed, hit_boundary_any)
}

/// Public: March in both directions from the start point and combine the results.
pub fn march_one_branch<T: FloatingPoint>(
    s: &NurbsSurface3D<T>,
    pl: &Plane<T>,
    uv0: (T, T),
    cfg: MarchConfig<T>,
) -> MarchResult<T> {
    // Check if the start point is on a boundary
    let ((umin, umax), (vmin, vmax)) = s.knots_domain();
    let mut start_active = Active::None;
    let tol_b = T::from_f64(10.0).unwrap() * cfg.tolerance;
    if (uv0.0 - umin).abs() <= tol_b {
        start_active = Active::UMin;
    } else if (uv0.0 - umax).abs() <= tol_b {
        start_active = Active::UMax;
    }
    if (uv0.1 - vmin).abs() <= tol_b {
        start_active = if start_active == Active::None {
            Active::VMin
        } else {
            Active::Corner
        };
    } else if (uv0.1 - vmax).abs() <= tol_b {
        start_active = if start_active == Active::None {
            Active::VMax
        } else {
            Active::Corner
        };
    }

    // + direction
    let (mut fwd, closed_f, hit_b_f) = march_one_direction(s, pl, uv0, start_active, cfg, T::one());
    // - direction
    let (mut bwd, closed_b, hit_b_b) =
        march_one_direction(s, pl, uv0, start_active, cfg, -T::one());

    // Combine: bwd is reversed, then remove the first (uv0) and concatenate
    if !bwd.is_empty() {
        bwd.reverse();
        if !bwd.is_empty() && !fwd.is_empty() && bwd.last().unwrap() == fwd.first().unwrap() {
            bwd.pop(); // Remove the first (uv0) if it is duplicated
        }
    }

    let mut uv = bwd;
    uv.append(&mut fwd);

    // Simple duplicate removal after concatenation (small movement)
    uv.dedup_by(|a, b| (a.x - b.x).abs() + (a.y - b.y).abs() < T::from_f64(1e-12).unwrap());

    MarchResult {
        uv,
        closed: closed_f || closed_b,
        hit_boundary: hit_b_f
            || hit_b_b
            || matches!(
                start_active,
                Active::UMin | Active::UMax | Active::VMin | Active::VMax
            ),
    }
}
