use std::{cell::RefCell, cmp::Ordering, rc::Rc};

use itertools::Itertools;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, U2, U3};

use crate::{
    boolean::{
        degeneracies::Degeneracy, has_parameter::HasParameter, node::Node, status::Status,
        vertex::Vertex,
    },
    curve::NurbsCurve,
    misc::{EndPoints, FloatingPoint},
    prelude::{CompoundCurveIntersection, Contains, HasIntersection, HasIntersectionParameter},
    region::{CompoundCurve, Region},
};

use super::operation::BooleanOperation;

/// Boolean operation for two curves.
/// Base algorithm reference: Efficient clipping of arbitrary polygons (https://www.inf.usi.ch/hormann/papers/Greiner.1998.ECO.pdf)
pub fn clip<'a, T: FloatingPoint, S, C, O: Clone>(
    subject: &'a S,
    clip: &'a C,
    operation: BooleanOperation,
    option: O,
    intersections: Vec<CompoundCurveIntersection<'a, T, U3>>,
) -> anyhow::Result<Vec<Region<T>>>
where
    S: Clone
        + Contains<T, U2, Option = O>
        + EndPoints<T, U2>
        + Into<CompoundCurve<T, U3>>
        + TrimRange<T, U3>,
    C: Clone
        + Contains<T, U2, Option = O>
        + EndPoints<T, U2>
        + Into<CompoundCurve<T, U3>>
        + TrimRange<T, U3>,
{
    let deg = intersections
        .into_iter()
        .map(|it| {
            let deg = Degeneracy::new(&it, it.a_curve(), it.b_curve());
            (it, deg)
        })
        .collect_vec();

    let (regular, deg): (Vec<_>, Vec<_>) = deg
        .into_iter()
        .partition(|(_, deg)| matches!(deg, Degeneracy::None));

    let regular = regular.into_iter().map(|(it, _)| it).collect_vec();
    let intersections = if regular.len() % 2 == 0 {
        regular
    } else {
        let max = deg.into_iter().max_by(|x, y| match (x.1, y.1) {
            (Degeneracy::Angle(x), Degeneracy::Angle(y)) => {
                x.partial_cmp(&y).unwrap_or(Ordering::Equal)
            }
            _ => Ordering::Equal,
        });
        match max {
            Some((max, _)) => [regular, vec![max]].concat(),
            _ => regular,
        }
    };

    anyhow::ensure!(
        intersections.len() % 2 == 0,
        "found odd number of intersections: {}",
        intersections.len()
    );

    let clip_contains_subject = clip.contains(&subject.first_point(), option.clone())?;
    let subject_contains_clip = subject.contains(&clip.first_point(), option.clone())?;

    let indexed = intersections.into_iter().enumerate().collect_vec();

    if indexed.is_empty() {
        let res = match (subject_contains_clip, clip_contains_subject) {
            (true, false) => match operation {
                BooleanOperation::Union => vec![Region::new(subject.clone().into(), vec![])],
                BooleanOperation::Intersection => {
                    vec![Region::new(clip.clone().into(), vec![])]
                }
                BooleanOperation::Difference => {
                    vec![Region::new(
                        subject.clone().into(),
                        vec![clip.clone().into()],
                    )]
                }
            },
            (false, true) => match operation {
                BooleanOperation::Union => vec![Region::new(clip.clone().into(), vec![])],
                BooleanOperation::Intersection => {
                    vec![Region::new(subject.clone().into(), vec![])]
                }
                BooleanOperation::Difference => {
                    vec![]
                }
            },
            (false, false) => match operation {
                BooleanOperation::Union => vec![
                    Region::new(subject.clone().into(), vec![]),
                    Region::new(clip.clone().into(), vec![]),
                ],
                BooleanOperation::Intersection => vec![],
                BooleanOperation::Difference => {
                    vec![Region::new(subject.clone().into(), vec![])]
                }
            },
            _ => {
                anyhow::bail!("Invalid case");
            }
        };
        return Ok(res);
    }

    // create linked list
    let mut a = indexed
        .iter()
        .sorted_by(|(_, i0), (_, i1)| i0.a_parameter().partial_cmp(&i1.a_parameter()).unwrap())
        .map(|(i, it)| {
            let (curve, p, t) = it.a();
            let v = Vertex::new(std::borrow::Borrow::borrow(curve), *p, *t);
            (*i, Rc::new(RefCell::new(Node::new(true, v))))
        })
        .collect_vec();

    let mut b = indexed
        .iter()
        .sorted_by(|(_, i0), (_, i1)| i0.b_parameter().partial_cmp(&i1.b_parameter()).unwrap())
        .map(|(i, it)| {
            let (curve, p, t) = it.b();
            let v = Vertex::new(std::borrow::Borrow::borrow(curve), *p, *t);
            (*i, Rc::new(RefCell::new(Node::new(false, v))))
        })
        .collect_vec();

    // connect neighbors
    a.iter_mut().for_each(|(index, node)| {
        if let Some((_, neighbor)) = b.iter().find(|(i, _)| i == index) {
            node.borrow_mut().set_neighbor(Rc::downgrade(neighbor));
        }
    });
    b.iter_mut().for_each(|(index, node)| {
        if let Some((_, neighbor)) = a.iter().find(|(i, _)| i == index) {
            node.borrow_mut().set_neighbor(Rc::downgrade(neighbor));
        }
    });

    // remove indices
    let a = a.into_iter().map(|(_, n)| n).collect_vec();
    let b = b.into_iter().map(|(_, n)| n).collect_vec();

    // connect cyclically
    [&a, &b].iter().for_each(|list| {
        list.iter()
            .cycle()
            .take(list.len() + 1)
            .collect_vec()
            .windows(2)
            .for_each(|w| {
                w[0].borrow_mut().set_next(Rc::downgrade(w[1]));
                w[1].borrow_mut().set_prev(Rc::downgrade(w[0]));
            });
    });

    // println!("clip contains subject: {}, subject contains clip: {}", clip_contains_subject, subject_contains_clip);

    let mut a_flag = !clip_contains_subject;
    let mut b_flag = !subject_contains_clip;

    a.iter().for_each(|list| {
        let mut node = list.borrow_mut();
        *node.status_mut() = if a_flag { Status::Enter } else { Status::Exit };
        a_flag = !a_flag;
    });

    b.iter().for_each(|list| {
        let mut node = list.borrow_mut();
        *node.status_mut() = if b_flag { Status::Enter } else { Status::Exit };
        b_flag = !b_flag;
    });

    match operation {
        BooleanOperation::Union | BooleanOperation::Difference => {
            // invert a status
            a.iter().for_each(|node| {
                node.borrow_mut().status_mut().invert();
            });
        }
        _ => {}
    }

    if operation == BooleanOperation::Union {
        // invert b status
        b.iter().for_each(|node| {
            node.borrow_mut().status_mut().invert();
        });
    }

    let mut regions: Vec<Region<T>> = vec![];

    fn non_visited<T: FloatingPoint>(
        node: Rc<RefCell<Node<Vertex<'_, T>>>>,
    ) -> Option<Rc<RefCell<Node<Vertex<'_, T>>>>> {
        let mut non_visited = node.clone();

        if non_visited.borrow().visited() {
            loop {
                let next = non_visited.borrow().next()?;
                non_visited = next;
                if Rc::ptr_eq(&node, &non_visited) || !non_visited.borrow().visited() {
                    break;
                }
            }
        }

        if non_visited.borrow().visited() {
            None
        } else {
            Some(non_visited)
        }
    }

    let subject = &a[0];

    /*
    println!(
        "a statues: {:?}",
        a.iter().map(|n| n.borrow().status()).collect_vec()
    );
    println!(
        "a parameters: {:?}",
        a.iter()
            .map(|n| n.borrow().vertex().parameter())
            .collect_vec()
    );
    println!(
        "b statues: {:?}",
        b.iter().map(|n| n.borrow().status()).collect_vec()
    );
    */

    loop {
        match non_visited(subject.clone()) {
            Some(start) => {
                let mut nodes = vec![];
                let mut current = start.clone();
                loop {
                    if current.borrow().visited() {
                        break;
                    }

                    current.borrow_mut().visit();
                    nodes.push(current.clone());

                    let node = current.borrow().clone();
                    let next = match node.status() {
                        Status::Enter => node.next(),
                        Status::Exit => node.prev(),
                        Status::None => todo!(),
                    };

                    if let Some(next) = next {
                        next.borrow_mut().visit();
                        nodes.push(next.clone());
                        if let Some(neighbor) = next.borrow().neighbor() {
                            current = neighbor.clone();
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                let mut spans = vec![];
                for chunk in nodes.chunks(2) {
                    if chunk.len() != 2 {
                        break;
                    }
                    let n0 = chunk[0].borrow();
                    let n1 = chunk[1].borrow();

                    /*
                    println!(
                        "{:?} {:?} -> {:?} {:?}",
                        n0.subject(),
                        n0.status(),
                        n1.subject(),
                        n1.status()
                    );
                    */

                    let params = match (n0.status(), n1.status()) {
                        (Status::Enter, Status::Exit) => {
                            (n0.vertex().parameter(), n1.vertex().parameter())
                        }
                        (Status::Exit, Status::Enter) => {
                            (n1.vertex().parameter(), n0.vertex().parameter())
                        }
                        _ => {
                            anyhow::bail!("Invalid status");
                        }
                    };

                    anyhow::ensure!(n0.subject() == n1.subject(), "Invalid condition");

                    let c0 = n0.vertex().curve();
                    let c1 = n1.vertex().curve();
                    if c0 == c1 {
                        let trimmed = c0.try_trim_range(params)?;
                        spans.extend(trimmed);
                    } else {
                        println!("params: {:?}", params);
                        if n0.subject() {
                        } else {
                        }
                    }
                }

                if !spans.is_empty() {
                    let region = Region::new(CompoundCurve::new(spans), vec![]);
                    regions.push(region);
                }
            }
            None => {
                break;
            }
        }
    }

    Ok(regions)
}

/// Trim curve by range parameters.
trait TrimRange<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    fn try_trim_range(&self, parameters: (T, T)) -> anyhow::Result<Vec<NurbsCurve<T, D>>>;
}

impl<T: FloatingPoint, D: DimName> TrimRange<T, D> for NurbsCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn try_trim_range(&self, parameters: (T, T)) -> anyhow::Result<Vec<NurbsCurve<T, D>>> {
        let (min, max) = (
            parameters.0.min(parameters.1),
            parameters.0.max(parameters.1),
        );
        let inside = parameters.0 < parameters.1;
        let curves = if inside {
            let (_, tail) = self.try_trim(min)?;
            let (head, _) = tail.try_trim(max)?;
            vec![head]
        } else {
            let (head, tail) = self.try_trim(min)?;
            let (_, tail2) = tail.try_trim(max)?;
            vec![tail2, head]
        };

        Ok(curves)
    }
}

impl<T: FloatingPoint, D: DimName> TrimRange<T, D> for CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn try_trim_range(&self, parameters: (T, T)) -> anyhow::Result<Vec<NurbsCurve<T, D>>> {
        todo!();
    }
}
