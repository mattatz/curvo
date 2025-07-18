use std::{cell::RefCell, cmp::Ordering, rc::Rc};

use itertools::Itertools;
use nalgebra::{Point2, U2, U3};

use crate::{
    boolean::{
        degeneracies::Degeneracy, has_parameter::HasParameter, node::Node, status::Status,
        vertex::Vertex,
    },
    curve::NurbsCurve2D,
    misc::{EndPoints, FloatingPoint},
    prelude::{CompoundCurveIntersection, Contains, HasIntersection, HasIntersectionParameter},
    region::{CompoundCurve, Region},
    trim::TrimRange,
};

use super::operation::BooleanOperation;

#[derive(Clone, Debug)]
#[allow(clippy::type_complexity)]
pub struct ClipDebugInfo<T>
where
    T: FloatingPoint,
{
    pub chunks: Vec<((Point2<T>, Status), (Point2<T>, Status))>,
    pub spans: Vec<NurbsCurve2D<T>>,
}

impl<T: FloatingPoint> Default for ClipDebugInfo<T> {
    fn default() -> Self {
        Self {
            chunks: vec![],
            spans: vec![],
        }
    }
}

#[allow(clippy::type_complexity)]
impl<T: FloatingPoint> ClipDebugInfo<T> {
    pub fn add_node_chunk(&mut self, chunk: ((Point2<T>, Status), (Point2<T>, Status))) {
        self.chunks.push(chunk);
    }

    pub fn node_chunks(&self) -> &Vec<((Point2<T>, Status), (Point2<T>, Status))> {
        &self.chunks
    }

    pub fn add_span(&mut self, span: NurbsCurve2D<T>) {
        self.spans.push(span);
    }

    pub fn spans(&self) -> &Vec<NurbsCurve2D<T>> {
        &self.spans
    }
}

/// A clip result
pub struct Clip<T: FloatingPoint> {
    regions: Vec<Region<T>>,
    info: ClipDebugInfo<T>,
}

impl<T: FloatingPoint> Clip<T> {
    pub fn new(regions: Vec<Region<T>>, info: ClipDebugInfo<T>) -> Self {
        Self { regions, info }
    }

    pub fn regions(&self) -> &Vec<Region<T>> {
        &self.regions
    }

    pub fn regions_mut(&mut self) -> &mut Vec<Region<T>> {
        &mut self.regions
    }

    pub fn into_regions(self) -> Vec<Region<T>> {
        self.regions
    }

    pub fn info(&self) -> &ClipDebugInfo<T> {
        &self.info
    }
}

/// Boolean operation for two curves.
/// Base algorithm reference: Efficient clipping of arbitrary polygons (https://www.inf.usi.ch/hormann/papers/Greiner.1998.ECO.pdf)
pub fn clip<'a, T: FloatingPoint, S, C, O: Clone>(
    subject: &'a S,
    clip: &'a C,
    operation: BooleanOperation,
    option: O,
    intersections: Vec<CompoundCurveIntersection<'a, T, U3>>,
) -> anyhow::Result<Clip<T>>
where
    S: Clone
        + Contains<C, Option = O>
        + EndPoints<T, U2>
        + Into<CompoundCurve<T, U3>>
        + TrimRange<T, U3>,
    C: Clone
        + Contains<S, Option = O>
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

    let clip_contains_subject = clip.contains(subject, option.clone())?;
    let subject_contains_clip = subject.contains(clip, option.clone())?;
    // println!("clip contains subject: {}, subject contains clip: {}", clip_contains_subject, subject_contains_clip);

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
        // return Ok(res);
        return Ok(Clip::new(res, Default::default()));
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

    let subject_head_node = &a[0];

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

    let mut info = ClipDebugInfo::default();

    while let Some(start) = non_visited(subject_head_node.clone()) {
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

            info.add_node_chunk((
                (*n0.vertex().position(), n0.status()),
                (*n1.vertex().position(), n1.status()),
            ));

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
                (Status::Enter, Status::Exit) => (n0.vertex().parameter(), n1.vertex().parameter()),
                (Status::Exit, Status::Enter) => (n1.vertex().parameter(), n0.vertex().parameter()),
                _ => {
                    anyhow::bail!("Invalid status");
                }
            };

            anyhow::ensure!(n0.subject() == n1.subject(), "Invalid condition");

            let trimmed = if n0.subject() {
                subject.try_trim_range(params)
            } else {
                clip.try_trim_range(params)
            }?;
            spans.extend(trimmed);
        }

        if !spans.is_empty() {
            spans.iter().for_each(|span| {
                info.add_span(span.clone());
            });

            // filter out degenerated spans
            let spans = spans
                .into_iter()
                .filter_map(|span| {
                    if span.degree() == 1 {
                        let mut pts = span.dehomogenized_control_points();
                        pts.dedup();
                        if pts.len() >= 2 {
                            Some(NurbsCurve2D::polyline(&pts, false))
                        } else {
                            None
                        }
                    } else {
                        Some(span)
                    }
                })
                .collect_vec();
            let region = Region::new(CompoundCurve::try_new(spans)?, vec![]);
            regions.push(region);
        }
    }

    Ok(Clip::new(regions, info))
}
