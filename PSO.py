# %%
from typing import Callable, List

import numpy as np

# %%
class PSO:
    def __init__(
        self,
        loss: Callable[[float, float], float],
        bounds: List[List[float]] = None,
        n_particles: int = 512,
    ) -> None:
        n_dimensions = 2  # could be an argument
        self.n_dimensions = n_dimensions
        self.n_particles = n_particles

        # problem specification
        if bounds is None:
            bounds = [[-np.inf, np.inf], [-np.inf, np.inf]]
        self.bounds = bounds
        self.loss = loss

        # particles
        self.positions = np.random.randn(n_particles, n_dimensions)
        self.velocities = np.random.randn(n_particles, n_dimensions)
        self.errors = np.zeros(n_particles)

        self.best_positions = self.positions
        self.best_loss = np.ones(n_particles) * np.inf

        self.global_best_position = self.best_positions[0]
        self.global_best_loss = np.inf

        # weights of adjustment components
        self.inertia = 0.4  # keep current heading
        self.dampening = 0.95  # reduce inertia over time
        self.cognitive = 1.9  # go towards personal best
        self.social = 1.9  # go towards global best

    def update_losses(self):
        """
        Evaluate fitness of all particles

        List comp used to allow any two argument function, if we use function 
        defined purely with math ops it can be replaced with much faster:
        self.loss(self.particles.T).T
        """
        self.errors = np.array([self.loss(x[0], x[1]) for x in self.positions])

    def update_best(self):
        # values for cognitive component
        better_positions = self.errors < self.best_loss
        self.best_positions = np.where(
            better_positions[:, np.newaxis], self.positions, self.best_positions
        )
        self.best_loss = np.where(better_positions, self.errors, self.best_loss)

        # values for social component
        global_best = np.argmin(self.best_loss)
        self.global_best_position = self.best_positions[global_best]
        self.global_best_loss = self.best_loss[global_best]

    def update_velocities(self):
        # stochastic multipliers
        r1 = np.random.random_sample((self.n_particles, 1))
        r2 = np.random.random_sample((self.n_particles, 1))

        # compute adjustment components
        inertial = self.inertia * self.velocities
        cognitive = self.cognitive * r1 * (self.best_positions - self.positions)
        social = self.social * r2 * (self.global_best_position - self.positions)

        self.inertia *= self.dampening
        self.velocities = inertial + cognitive + social

    def update_positions(self):
        self.positions += self.velocities

        for dim in range(self.n_dimensions):
            self.positions[:, dim] = np.clip(self.positions[:, dim], *self.bounds[dim])

    def step(self):
        self.update_losses()
        self.update_best()
        self.update_velocities()
        self.update_positions()

    def fit(self, n_steps: int = 1000, epsilon: float = 1e-3, verbose: bool = False):
        for i in range(n_steps):
            self.step()

            if verbose:
                self.print_status(i)

            if np.sum(np.abs(self.velocities)) < epsilon:
                # particles no longer moving, stop early
                break
        return i, self.global_best_position, self.global_best_loss

    def print_status(self, i):
        print(
            f"STEP {i}: "
            f"global ε = {self.global_best_loss}, "
            f"global p = {self.global_best_position}, "
            f"mean ε = {np.mean(self.best_loss)}, "
            f"max ε = {np.max(self.best_loss)}, "
            f"mean v = {np.mean(self.velocities, axis=0)}, "
            f"mean p = {np.mean(self.positions, axis=0)}"
        )


if __name__ == "__main__":
    # %%
    cost = (
        lambda x, y: np.sin(x)
        + np.cos(y)
        + 1.5 * np.sin(2 * np.pi * 10 * x + 0.42)
        + 4 * x * np.exp(-(x ** 2) - y ** 2)
    )

    # %%
    # Just get the results
    opt = PSO(cost, n_particles=512, bounds=[[-5, 5], [-5, 5]])
    steps, position, loss = opt.fit(verbose=True)
    print(
        f"Optimization done in {steps} steps, solution: {position}, final loss: {loss}"
    )

    # %%
    # Visualize 64 steps
    import plotly.graph_objects as go

    STEPS = 64
    SECONDS = 30

    opt = PSO(cost, n_particles=512, bounds=[[-5, 5], [-5, 5]])
    frames = []
    for i in range(STEPS):
        opt.step()

        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=opt.positions[:, 0],
                        y=opt.positions[:, 1],
                        z=opt.best_loss,
                        mode="markers",
                        marker={"colorscale": "Viridis", "color": opt.best_loss},
                    )
                ],
            )
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=opt.positions[:, 0],
                y=opt.positions[:, 1],
                z=opt.best_loss,
                mode="markers",
                marker={"colorscale": "Viridis", "color": opt.best_loss},
            )
        ],
        frames=frames,
        layout=go.Layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": SECONDS * 1000 // STEPS,
                                        "redraw": True,
                                    },
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "type": "buttons",
                }
            ],
        ),
    )
    fig.update_layout(
        scene_aspectmode="cube",
        scene_xaxis=dict(range=opt.bounds[0], autorange=False),
        scene_yaxis=dict(range=opt.bounds[1], autorange=False),
        scene_zaxis=dict(range=[-4, 4], autorange=False),
    )
    fig.write_html("pso.html", auto_play=False)
    # fig.show()

    # %%
