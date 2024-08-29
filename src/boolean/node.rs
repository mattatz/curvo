use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

use crate::misc::FloatingPoint;

use super::{status::Status, vertex::Vertex};

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

    pub fn set_prev(&mut self, prev: Weak<RefCell<Self>>) {
        self.prev = Some(prev);
    }

    pub fn next(&self) -> Option<Rc<RefCell<Self>>> {
        self.next.clone().and_then(|n| n.upgrade())
    }

    pub fn set_next(&mut self, next: Weak<RefCell<Self>>) {
        self.next = Some(next);
    }

    pub fn neighbor(&self) -> Option<Rc<RefCell<Self>>> {
        self.neighbor.clone().and_then(|n| n.upgrade())
    }

    pub fn set_neighbor(&mut self, neighbor: Weak<RefCell<Self>>) {
        self.neighbor = Some(neighbor);
    }

    pub fn status(&self) -> Status {
        self.status
    }

    pub fn status_mut(&mut self) -> &mut Status {
        &mut self.status
    }

    pub fn visited(&self) -> bool {
        self.visited
    }

    pub fn visit(&mut self) {
        self.visited = true;
    }
}
