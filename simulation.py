import logging
from typing import Optional
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
import base64

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from copy import deepcopy


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
        oscillator_masses: list[int],
        springs_current_lens: Optional[list[int]] = None,
        spring_constants: Optional[list[int]] = None,
        elastic_collisions: bool = False,
        damping: float = 0.0,
    ):
        self.num_oscillators = len(oscillator_masses)
        self.num_springs = self.num_oscillators + 1
        self.spring_default_lens = [OscillatorsSimulation.SPRING_BASE_LEN,] * self.num_springs

        if spring_constants is None:
            spring_constants = [
                self.SPRING_BASE_K,
            ] * self.num_springs

        if springs_current_lens is None:
            springs_current_lens = [
                self.SPRING_BASE_LEN,
            ] * self.num_springs
        elif len(springs_current_lens) != self.num_springs:
            raise ValueError(
                f"Different numbers of string constants and strings lens. Constants: {self.num_springs}. Lens: {len(springs_current_lens)}"
            )
        elif sum(springs_current_lens) != self.SPRING_BASE_LEN * self.num_springs:
            raise ValueError(
                "Springs lengths should sum up to the width between the walls."
                f"Sum of provided lengths: {sum(springs_current_lens)}."
                f"Expected sum: {self.SPRING_BASE_LEN*self.num_springs}"
            )

        # TODO: probably can be deleted - since we supposed total length is constant
        self.total_length = sum(springs_current_lens)

        if self.num_springs != self.num_oscillators + 1:
            raise ValueError(
                f"Invalid number of springs or oscillators. Springs number: {self.num_springs}. Oscillators number: {self.num_oscillators}"
            )

        self.oscillator_masses = np.array(oscillator_masses, float)
        self.spring_constants = np.array(spring_constants, float)
        
        self.fig, axes = plt.subplots(5, 1, figsize=(10, 20))  # Adjust the layout as needed
        self.ax, self.phase_ax, self.force_ax, self.velocity_ax, self.position_ax = axes

        # Parameters
        self.time_step = 0.005  # Time step

        # Initial conditions
        x = np.cumsum(springs_current_lens[:-1], dtype=float)  # X Coordinates
        v = np.zeros(self.num_oscillators)  # Velocities

        self.current_state = self.OscillatorsState(
            x, v, np.zeros(self.num_springs, float)
        )
        self.break_states_generator = False

        self.elastic_collisions = False
        assert damping >= 0, "Can't get negative damping"
        self.damping = damping
    
    

    def get_spring_lens(self, state: OscillatorsState = None):
        if state is None:
            state = self.current_state
        return np.concatenate(
            [state.oscillators_x, np.array([self.total_length], float)]
        ) - np.concatenate(
            [
                np.array([OscillatorsSimulation.LEFT_WALL_X_CORD], float),
                state.oscillators_x,
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
            self.spring_default_lens[spring_id] - self.get_spring_lens()[spring_id]
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

        # for the springs on the walls we utilize all their force on the oscillators
        # for others, force is distributed between both ends, therefore we divide by 2
        if oscillator_id != 0:
            left_spring_force *= 0.5
        if oscillator_id != (self.num_oscillators - 1):
            right_spring_force *= 0.5

        # Supposition: X axis starts at the left wall and is positive on the right side from it
        oscillator_force = right_spring_force - left_spring_force

        # ref: https://tomgrad.fizyka.pw.edu.pl/foi/01-drgania/#/1
        damping_force = self.current_state.oscillators_v[oscillator_id] * self.damping
        oscillator_force -= damping_force
        return oscillator_force

    @staticmethod
    def get_indices_of_collisions(x: np.ndarray) -> list[np.ndarray]:
        """
        Calculates indices of collisions. Math:
        1.  calculates max from the left, min from the right for each element
            x = [1 2 5 3 4 2 8], lmax = [1 2 5 5 5 5 8], rmin = [1 2 2 2 2 2 8]
            mask = [T T F F F F T]
        2.  catches all instances of masses being in the same spot as other masses
            e.g. x = [1 2 5 3 4 2 8] → uni, cts = [1 2 3 4 5 8], [1 2 1 1 1 1]
            mask = [T F T T T F T]
        3.  combine with above: [T F F F F F T]
            where F = collision
            indices = [1, 2, 3, 4, 5]
        4.  Now group the collisions in group of collisions
            e.g x = [0, 0, 1, 2, 2, 3, 3] → 3 collisions, [0, 0], [2, 2] and [3, 3],
            indices = [0, 1, 3, 4, 5, 6]. For that we use the 1. step once again,
            but on the x[indices] and then split when lmax < rmin
            we get [0, 0], [2, 2], [3, 3]
        Returns list of np.ndarrays, each array representing indices
        """
        mask = np.ones_like(x, dtype=bool)

        # 1
        left_max = np.maximum.accumulate(x)
        right_min = np.minimum.accumulate(x[::-1])[::-1]

        mask = mask & ((x >= left_max) & (x <= right_min))

        # 2 & 3
        uni, cnts = np.unique(x, return_counts=True)
        mask = mask & np.isin(x, uni[cnts == 1])
        indices = np.arange(len(x))[~mask]

        # 4
        left_max = np.maximum.accumulate(x[indices])
        right_min = np.minimum.accumulate(x[indices][::-1])[::-1]

        groups = np.split(
            indices, np.arange(1, len(indices))[left_max[:-1] < right_min[1:]]
        )
        # should not happen
        assert not np.array_equal(groups, [[]]), "No collisions found"
        return groups

    def calc_not_elastic_collisions(self, collisions: list[np.ndarray]) -> None:
        """
        Calculates velocities of all masses being part of all nonelastic collisions
        Math: (for each collision): calculate total momentum, then calculate the velocity
        of the "new" mass consisting of all the masses and substitute.
        Also substitute the positions of all the masses to one position (in order to
        avoid mess like mass on the 4th place becoming mass on the 5th place and vice
        versa if such collision occurrs)
        """
        for collision in collisions:
            momentum = np.sum(
                self.oscillator_masses[collision]
                * self.current_state.oscillators_v[collision]
            )  # momentum
            v = momentum / np.sum(self.oscillator_masses[collision])  # group velocity
            self.current_state.oscillators_v[collision] = v
            self.current_state.oscillators_x[collision] = np.mean(
                self.current_state.oscillators_x[collision]
            )  # group new position

    def calc_elastic_collisions(self, collisions: list[np.ndarray]) -> None:
        """
        ONLY SUPPORTS 2 MASS COLLISION
        Calculates velocities of all masses being part of all elastic collisions
        Math: (for each collision): calc velocity using math from wikipedia:
        https://en.wikipedia.org/wiki/Elastic_collision
        Also substitute the positions of all the masses to one position (in order to
        avoid mess like mass on the 4th place becoming mass on the 5th place and vice
        versa if such collision occurrs)
        """
        for collision in collisions:
            assert len(collision) == 2, "ONLY SUPPORTS 2 MASS COLLISION"

            m1, m2 = self.oscillator_masses[collision]
            u1, u2 = self.current_state.oscillators_v[collision]
            # math from wiki

            v1 = u1 * (m1 - m2) / (m1 + m2) + u2 * 2 * m2 / (m1 + m2)
            v2 = -u2 * (m1 - m2) / (m1 + m2) + u2 * 2 * m1 / (m1 + m2)

            self.current_state.oscillators_v[collision] = [v1, v2]
            self.current_state.oscillators_x[collision] = np.mean(
                self.current_state.oscillators_x[collision]
            )  # group new position

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
            col_groups = self.get_indices_of_collisions(tmp_x)
            if self.elastic_collisions:
                self.calc_elastic_collisions(col_groups)
            else:
                self.calc_not_elastic_collisions(col_groups)
        else:
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

    # Mockup data for demonstration
    def get_plots(self):
        
        # Save plot to a BytesIO object
        img = BytesIO()
        self.fig.savefig(img, format='png')
        img.seek(0)

        # Encode the image to base64 for embedding in HTML
        return base64.b64encode(img.getvalue()).decode('utf-8')

    def create_animation(self, states_number=1000):
        states_gen = self.gen_states(states_number)

        states = []
        while True:
            try:
                new_state = deepcopy(next(states_gen))
            except StopIteration:
                break
            states.append(new_state)

        all_xs = np.array([state.oscillators_x for state in states])
        all_fs = np.array([state.springs_f for state in states])
        all_vs = np.array([state.oscillators_v for state in states])
        fmax = np.max(all_fs)
        fmin = np.min(all_fs)

        # TODO: theoretical limits
        # fmax_squash = max(s_len * s_k for s_len, s_k in zip(self.spring_default_lens, self.spring_constants))
        # fmax_stretch = max(self.total_length * s_k for s_k in self.spring_constants)
        # fmax = max((fmax_stretch, fmax_squash))
        # fmin = 0
        
        self.phase_data = []
        self.force_data = []
        self.velocity_data = []
        self.position_data = []
        self.phase_ax.clear()
        self.force_ax.clear()
        self.velocity_ax.clear()
        self.position_ax.clear()

        def animate(i):
            xs = all_xs[i]
            fs = all_fs[i]
            springs_lens = self.get_spring_lens(states[i])

            self.ax.clear()
            self.ax.set_ylim(-1, 1)
            self.ax.set_xlim(0, self.total_length)
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

            self.ax.scatter(xs, np.zeros_like(xs), c="b")
            for x, num in zip(xs, np.arange(self.num_oscillators)):
                self.ax.annotate(
                    f"{num}",
                    (x, 0),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                )

            # Update phase plot
            phase = np.arctan2(all_xs[i], all_vs[i])
            
            self.phase_ax.plot(np.arange(self.num_oscillators), phase, 'ro-')
            self.phase_ax.set_title('Oscillator Phases')

            # Update force plot

            self.force_ax.plot(np.arange(self.num_springs), all_fs[i], 'bo-')
            self.force_ax.set_title('Forces on Oscillators')

            # Update velocity plot
            
            self.velocity_ax.plot(np.arange(self.num_oscillators), all_vs[i], 'go-')
            self.velocity_ax.set_title('Velocities of Oscillators')

            # Update position plot
            
            self.position_ax.plot(np.arange(self.num_oscillators), all_xs[i], 'mo-')
            self.position_ax.set_title('Positions of Oscillators')


        ani = FuncAnimation(self.fig, animate, frames=range(0, states_number, 100))
        # TODO: remove saving - it will be faster
        ani.save("animation.gif", writer="imagemagick")
        return ani
