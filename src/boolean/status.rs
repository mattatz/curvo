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
