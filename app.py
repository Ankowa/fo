from dataclasses import dataclass
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, Response
from matplotlib.animation import FuncAnimation

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/animation")
def animation():
    os = OscillatorsSimulation([2, 2], [2, 2, 2], [1, 4, 1])
    ani = os.create_animation()
    return Response(ani.to_jshtml(), content_type="text/html")


class OscillatorsSimulation:
    SPRING_BASE_LEN = 20
    SPRING_BASE_K = 1
    LEFT_WALL_X_CORD = 0

    @dataclass
    class OscillatorsState:
        oscillators_x: np.ndarray
        oscillators_v: np.ndarray
        springs_f: np.ndarray

    def __init__(
        self,
        oscillator_masses,
        springs_default_lens,
        springs_current_lens=None,
        spring_constants=None,
        elastic_collisions=False,
        damping=0,
    ):
        assert damping >= 0, "Can't get negative damping"

        self.num_springs = len(springs_default_lens)
        self.num_oscillators = len(oscillator_masses)

        # TODO: extend support - if added no need for equilibrium springs lens
        if spring_constants is None:
            spring_constants = [
                self.SPRING_BASE_K,
            ] * self.num_springs
        if len(set(spring_constants)) != 1:
            raise NotImplementedError(
                "When calculating equilibrium point from ks is added, different values of springs ks will be allowed"
            )

        if springs_current_lens is None:
            springs_current_lens = [
                self.SPRING_BASE_LEN,
            ] * self.num_springs
        elif len(springs_current_lens) != self.num_springs:
            raise ValueError(
                f"Different numbers of string constants and strings lens. Constants: {self.num_springs}. Lens: {len(springs_current_lens)}"
            )

        self.total_length = sum(springs_current_lens)

        if self.num_springs != self.num_oscillators + 1:
            raise ValueError(
                f"Invalid number of springs or oscillators. Springs number: {self.num_springs}. Oscillators number: {self.num_oscillators}"
            )

        self.oscillator_masses = np.array(oscillator_masses, float)
        self.spring_constants = np.array(spring_constants, float)
        self.spring_default_lens = np.array(springs_default_lens, float)

        self.fig, self.ax = plt.subplots()

        # Parameters
        self.time_step = 0.005  # Time step

        # Initial conditions
        x = np.cumsum(springs_current_lens[:-1], dtype=float)  # X Coordinates
        v = np.zeros(self.num_oscillators)  # Velocities

        self.current_state = self.OscillatorsState(
            x, v, np.zeros(self.spring_default_lens.shape, float)
        )
        self.break_states_generator = False

        self.elastic_collisions = elastic_collisions

    @cached_property
    def get_spring_lens(self):
        return np.concatenate(
            [self.current_state.oscillators_x, np.array([self.total_length], float)]
        ) - np.concatenate(
            [
                np.array([OscillatorsSimulation.LEFT_WALL_X_CORD], float),
                self.current_state.oscillators_x,
            ]
        )

    @staticmethod
    def get_neighbor_springs_ids(oscillator_id):
        return oscillator_id, oscillator_id + 1

    @staticmethod
    def get_prev_oscillator_id(spring_id):
        return spring_id - 1 if spring_id else None

    @staticmethod
    def get_color(f, fmax, fmin):
        if f > 0:
            return (1 - f / fmax, 1 - 0.5 * (f / fmax), 0)
        else:
            return (1, 1 - f / fmin, 0)

    def get_spring_force(self, spring_id):
        # Force is negative if spring is currently shorter than by default ->
        # It requires to ALWAYS suppose (when designing forces equations),
        # that forces vectors are directed as if springs were stretched out
        spring_len_diff = (
            self.spring_default_lens[spring_id] - self.get_spring_lens[spring_id]
        )
        spring_force = -1 * self.spring_constants[spring_id] * spring_len_diff
        return spring_force

    def get_oscillator_force(self, oscillator_id):
        left_spring_id, right_spring_id = self.get_neighbor_springs_ids(oscillator_id)
        left_spring_force = self.get_spring_force(left_spring_id)
        right_spring_force = self.get_spring_force(right_spring_id)
        # Only necessary for the first spring of all - could be improved
        self.current_state.springs_f[left_spring_id] = left_spring_force
        self.current_state.springs_f[right_spring_id] = right_spring_force
        # Supposition: X axis starts at the left wall and is positive on the right side from it
        oscillator_force = right_spring_force - left_spring_force
        return oscillator_force

    @staticmethod
    def get_indices_of_collisions(x: np.ndarray) -> np.ndarray:
        """
        Calculates indices of collisions. Math:
        1.  calculates max from the left, min from the right for each element
            x = [1 2 5 3 4 2 8], lmax = [1 2 5 5 5 5 8], rmin = [1 2 2 2 2 2 8]
            expr = [T T F F F F T]
        2.  catches all instances of masses being in the same spot as other masses
            e.g. x = [1 2 5 3 4 2 8] → uni, cts = [1 2 3 4 5 8], [1 2 1 1 1 1]
            expr → [T F T T T F T]
        3.  combine with above: [T F F F F F T]
            where F = collision
        """
        indices = np.ones_like(x, dtype=bool)

        # 1
        left_max = np.maximum.accumulate(x)
        right_min = np.minimum.accumulate(x[::-1])[::-1]

        indices = indices & ((x >= left_max) & (x <= right_min))

        # 2 & 3
        uni, cnts = np.unique(x, return_counts=True)
        indices = indices & np.isin(x, uni[cnts == 1])

        return np.arange(len(x))[~indices]

    def not_elastic_collisions(self, indices: np.ndarray) -> None:
        """
        Resets velocities of all elements of collisions to 0
        """
        self.current_state.oscillators_v[indices] = 0

    def elastic_collisions(self, indices: np.ndarray) -> None:
        """
        TODO here math for that
        """
        ...

    def calc_next_state(self):
        oscillators_a = np.array(
            [
                self.get_oscillator_force(oscillator_id) / oscillator_mass
                for oscillator_id, oscillator_mass in enumerate(self.oscillator_masses)
            ],
            float,
        )
        self.current_state.oscillators_v += oscillators_a * self.time_step
        tmp_x = (
            self.current_state.oscillators_x
            + self.current_state.oscillators_v * self.time_step
        )

        if ~np.all(
            tmp_x[:-1] < tmp_x[1:]
        ):  # masses' x coordinates are not sorted → collisions
            col_indices = self.get_indices_of_collisions(tmp_x)
            if self.elastic_collisions:
                self.elastic_collisions(col_indices)
            else:
                self.not_elastic_collisions(col_indices)

        self.current_state.oscillators_x += (
            self.current_state.oscillators_v * self.time_step
        )

        return self.current_state

    def gen_states(self, states_number):
        for state_nr in range(states_number):
            yield self.calc_next_state()

    def gen_infinite_states(self):
        while True:
            if self.break_states_generator:
                break
            yield self.calc_next_state()

    def create_animation(self, states_number=100):
        states_gen = self.gen_states(states_number)
        fs_gen = map(lambda state: state.springs_f, states_gen)
        fmax = max(np.max(f) for f in fs_gen)

        states_gen = self.gen_states(states_number)
        fs_gen = map(lambda state: state.springs_f, states_gen)
        fmin = max(np.min(f) for f in fs_gen)

        states_gen = self.gen_states(states_number)

        # TODO: theoretical limits
        # fmax_squash = max(s_len * s_k for s_len, s_k in zip(self.spring_default_lens, self.spring_constants))
        # fmax_stretch = max(self.total_length * s_k for s_k in self.spring_constants)
        # fmax = max((fmax_stretch, fmax_squash))
        # fmin = 0

        def animate(state):
            xs = state.oscillators_x
            fs = state.springs_f
            springs_lens = self.get_spring_lens

            self.ax.clear()
            self.ax.set_ylim(-1, 1)
            self.ax.set_xlim(0, 10)
            self.ax.scatter(xs, np.zeros_like(xs), c="b")
            self.ax.plot([0, xs[0]], [0, 0], c=self.get_color(fs[0], fmax, fmin))
            self.ax.plot(
                [xs[-1], self.total_length],
                [0, 0],
                c=self.get_color(fs[-1], fmax, fmin),
            )

            for spring_id, spring_len in enumerate(springs_lens[1:-1], 1):
                prev_oscillator_id = self.get_prev_oscillator_id(spring_id)
                prev_oscillator_x = xs[prev_oscillator_id]
                self.ax.plot(
                    [prev_oscillator_x, prev_oscillator_x + spring_len],
                    [0, 0],
                    c=self.get_color(fs[spring_id], fmax, fmin),
                )

            return self.current_state

        ani = FuncAnimation(self.fig, animate, frames=states_gen)
        return ani
        # ani.save('notebooks/plots/01-general-spring.gif', writer='pillow')
        # plt.close()


if __name__ == "__main__":
    app.run(debug=True)
