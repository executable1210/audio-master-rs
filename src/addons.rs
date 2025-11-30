pub trait VecMove {
    fn move_index(&mut self, from: usize, to: usize);
}

impl<T> VecMove for Vec<T> {
    fn move_index(&mut self, from: usize, to: usize) {
        let len = self.len();
        assert!(from < len && to < len, "index out of bounds");

        if from == to {
            return;
        }

        if from < to {
            self[from..=to].rotate_left(1);
        } else {
            self[to..=from].rotate_right(1);
        }
    }
}
