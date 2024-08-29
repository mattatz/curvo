use std::{
    cell::RefCell,
    fmt::Display,
    rc::{Rc, Weak},
};

use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, ComplexField, Const, DefaultAllocator, DimName, Point2, Vector2,
};

use crate::{
    curve::NurbsCurve,
    misc::{FloatingPoint, Line},
    prelude::{Contains, CurveIntersection, CurveIntersectionSolverOptions},
    region::{CompoundCurve, Region},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOperation {
    Union,
    Intersection,
    Difference,
}

impl Display for BooleanOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BooleanOperation::Union => write!(f, "Union"),
            BooleanOperation::Intersection => write!(f, "Intersection"),
            BooleanOperation::Difference => write!(f, "Difference"),
        }
    }
}

/// A trait for boolean operations.
pub trait Boolean<T> {
    type Output;
    type Option;

    fn union(&self, other: T, option: Self::Option) -> Self::Output;
    fn intersection(&self, other: T, option: Self::Option) -> Self::Output;
    fn difference(&self, other: T, option: Self::Option) -> Self::Output;
    fn boolean(&self, operation: BooleanOperation, other: T, option: Self::Option) -> Self::Output;
}

impl<'a, T: FloatingPoint + ArgminFloat> Boolean<&'a NurbsCurve<T, Const<3>>>
    for NurbsCurve<T, Const<3>>
where
    DefaultAllocator: Allocator<Const<3>>,
{
    // type Output = anyhow::Result<Vec<Region<T>>>;
    type Output = anyhow::Result<(Vec<Region<T>>, Vec<Node<T>>)>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    fn union(&self, other: &'a NurbsCurve<T, Const<3>>, option: Self::Option) -> Self::Output {
        self.boolean(BooleanOperation::Union, other, option)
    }

    fn intersection(
        &self,
        other: &'a NurbsCurve<T, Const<3>>,
        option: Self::Option,
    ) -> Self::Output {
        self.boolean(BooleanOperation::Intersection, other, option)
    }

    fn difference(&self, other: &'a NurbsCurve<T, Const<3>>, option: Self::Option) -> Self::Output {
        self.boolean(BooleanOperation::Difference, other, option)
    }

    fn boolean(
        &self,
        operation: BooleanOperation,
        other: &'a NurbsCurve<T, Const<3>>,
        option: Self::Option,
    ) -> Self::Output {
        // let a_knot_domain = self.knots_domain();
        // let b_knot_domain = other.knots_domain();
        let parameter_eps = T::from_f64(1e-3).unwrap();
        let tangent_threshold = T::one() - T::from_f64(1e-2).unwrap();

        let intersections = self.find_intersections(other, option.clone())?;
        // println!("intersections: {}", intersections.len());

        let intersections = intersections
            .into_iter()
            .filter(|it| {
                let a0 = self.point_at(it.a().1 - parameter_eps);
                let a1 = self.point_at(it.a().1 + parameter_eps);
                let la = Line::new(a0, a1);
                let b0 = other.point_at(it.b().1 - parameter_eps);
                let b1 = other.point_at(it.b().1 + parameter_eps);
                let lb = Line::new(b0, b1);
                let intersected = la.intersects(&lb);
                let dot =
                    ComplexField::abs(la.tangent().normalize().dot(&lb.tangent().normalize()));
                // println!("intersected: {}, dot: {}, ({}, {}) - ({}, {})", intersected, dot, a0, a1, b0, b1);
                intersected && dot < tangent_threshold
            })
            .enumerate()
            .collect_vec();

        println!("intersections: {}", intersections.len());

        if intersections.is_empty() {
            anyhow::bail!("Todo: no intersections case");
        }

        // create linked list
        let mut a = intersections
            .iter()
            .sorted_by(|(_, i0), (_, i1)| i0.a().1.partial_cmp(&i1.a().1).unwrap())
            .map(|(i, it)| (*i, Rc::new(RefCell::new(Node::new(true, it.a().into())))))
            .collect_vec();

        let mut b = intersections
            .iter()
            .sorted_by(|(_, i0), (_, i1)| i0.b().1.partial_cmp(&i1.b().1).unwrap())
            .map(|(i, it)| (*i, Rc::new(RefCell::new(Node::new(false, it.b().into())))))
            .collect_vec();

        // connect neighbors
        a.iter_mut().for_each(|(index, node)| {
            b.iter().find(|(i, _)| i == index).map(|(_, neighbor)| {
                node.borrow_mut().neighbor = Some(Rc::downgrade(neighbor));
            });
        });
        b.iter_mut().for_each(|(index, node)| {
            a.iter().find(|(i, _)| i == index).map(|(_, neighbor)| {
                node.borrow_mut().neighbor = Some(Rc::downgrade(neighbor));
            });
        });

        // remove indices
        let a = a.into_iter().map(|(_, n)| n).collect_vec();
        let b = b.into_iter().map(|(_, n)| n).collect_vec();

        [&a, &b].iter().for_each(|list| {
            list.iter()
                .cycle()
                .take(list.len() + 1)
                .collect_vec()
                .windows(2)
                .for_each(|w| {
                    w[0].borrow_mut().next = Some(Rc::downgrade(w[1]));
                    w[1].borrow_mut().prev = Some(Rc::downgrade(w[0]));
                });
        });

        let mut a_flag = !other.contains(&self.point_at(self.knots_domain().0), option.clone())?;
        let mut b_flag = !self.contains(&other.point_at(other.knots_domain().0), option.clone())?;
        println!("a flag: {}, b flag: {}", a_flag, b_flag);

        a.iter().for_each(|list| {
            let mut node = list.borrow_mut();
            node.status = if a_flag { Status::Enter } else { Status::Exit };
            a_flag = !a_flag;
        });

        b.iter().for_each(|list| {
            let mut node = list.borrow_mut();
            node.status = if b_flag { Status::Enter } else { Status::Exit };
            b_flag = !b_flag;
        });

        match operation {
            BooleanOperation::Union | BooleanOperation::Difference => {
                // invert a status
                a.iter().for_each(|node| {
                    node.borrow_mut().status.invert();
                });
            }
            _ => {}
        }

        match operation {
            BooleanOperation::Union => {
                // invert b status
                b.iter().for_each(|node| {
                    node.borrow_mut().status.invert();
                });
            }
            _ => {}
        }

        // Efficient clipping of arbitrary polygons
        // https://www.inf.usi.ch/hormann/papers/Greiner.1998.ECO.pdf

        let mut regions = vec![];

        let non_visited = |node: Rc<RefCell<Node<T>>>| -> Option<Rc<RefCell<Node<T>>>> {
            let mut non_visited = node.clone();

            if non_visited.borrow().visited {
                loop {
                    let next = non_visited.borrow().next()?;
                    non_visited = next;
                    if Rc::ptr_eq(&node, &non_visited) || !non_visited.borrow().visited {
                        break;
                    }
                }
            }

            if non_visited.borrow().visited {
                None
            } else {
                Some(non_visited)
            }
        };

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
                    // println!("start: {:?}", start.borrow().vertex.parameter());
                    let mut nodes = vec![];
                    let mut current = start.clone();
                    loop {
                        if current.borrow().visited {
                            break;
                        }

                        // let forward = matches!(current.borrow().status(), Status::Enter);
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

                    // println!("nodes: {:?}", nodes.len());
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
                                println!("Invalid status");
                                break;
                            }
                        };

                        match (n0.subject(), n1.subject()) {
                            (true, true) => {
                                let trimmed = try_trim(self, params)?;
                                spans.extend(trimmed);
                            }
                            (false, false) => {
                                let trimmed = try_trim(other, params)?;
                                spans.extend(trimmed);
                            }
                            _ => {
                                println!("subject & clip case");
                            }
                        }
                    }

                    let region = Region::new(CompoundCurve::new(spans), vec![]);
                    regions.push(region);
                }
                None => {
                    break;
                }
            }
        }

        Ok((regions, a.iter().map(|n| n.borrow().clone()).collect_vec()))
    }
}

fn try_trim<T: FloatingPoint, D: DimName>(
    curve: &NurbsCurve<T, D>,
    parameters: (T, T),
) -> anyhow::Result<Vec<NurbsCurve<T, D>>>
where
    DefaultAllocator: Allocator<D>,
{
    let (min, max) = (
        parameters.0.min(parameters.1),
        parameters.0.max(parameters.1),
    );
    let inside = if parameters.0 < parameters.1 {
        true
    } else {
        false
    };
    let curves = if inside {
        let (_, tail) = curve.try_trim(min)?;
        let (head, _) = tail.try_trim(max)?;
        vec![head]
    } else {
        let (head, tail) = curve.try_trim(min)?;
        let (_, tail2) = tail.try_trim(max)?;
        vec![tail2, head]
    };

    Ok(curves)
}

#[derive(Debug, Clone)]
pub struct Vertex<T: FloatingPoint> {
    position: Point2<T>,
    parameter: T,
}

impl<T: FloatingPoint> Vertex<T> {
    pub fn new(position: Point2<T>, parameter: T) -> Self {
        Self {
            position,
            parameter,
        }
    }

    pub fn position(&self) -> &Point2<T> {
        &self.position
    }

    pub fn parameter(&self) -> T {
        self.parameter
    }
}

impl<'a, T: FloatingPoint> From<&'a (Point2<T>, T)> for Vertex<T> {
    fn from(v: &'a (Point2<T>, T)) -> Self {
        Self {
            position: v.0,
            parameter: v.1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Node<T: FloatingPoint> {
    subject: bool,
    vertex: Vertex<T>,
    prev: Option<Weak<RefCell<Node<T>>>>,
    next: Option<Weak<RefCell<Node<T>>>>,
    neighbor: Option<Weak<RefCell<Node<T>>>>,
    status: Status,
    visited: bool,
}

impl<T: FloatingPoint> Node<T> {
    pub fn new(subject: bool, vertex: Vertex<T>) -> Self {
        Self {
            subject,
            vertex,
            prev: None,
            next: None,
            neighbor: None,
            status: Status::None,
            visited: false,
        }
    }

    pub fn subject(&self) -> bool {
        self.subject
    }

    pub fn vertex(&self) -> &Vertex<T> {
        &self.vertex
    }

    pub fn prev(&self) -> Option<Rc<RefCell<Self>>> {
        self.prev.clone().and_then(|p| p.upgrade())
    }

    pub fn next(&self) -> Option<Rc<RefCell<Self>>> {
        self.next.clone().and_then(|n| n.upgrade())
    }

    pub fn neighbor(&self) -> Option<Rc<RefCell<Self>>> {
        self.neighbor.clone().and_then(|n| n.upgrade())
    }

    pub fn status(&self) -> Status {
        self.status
    }

    pub fn visit(&mut self) {
        self.visited = true;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    None,
    Enter,
    Exit,
}

impl Status {
    pub fn invert(&mut self) {
        *self = match self {
            Status::Enter => Status::Exit,
            Status::Exit => Status::Enter,
            Status::None => Status::None,
        };
    }
}
