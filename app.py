from flask import Flask, render_template, request
from simulation import OscillatorsSimulation

app = Flask(__name__)
N_STATES = 1  # quick start


@app.route("/", methods=["GET", "POST"])
def index():
    print("______________________________________________________")
    if request.method == "POST":
        number_of_oscillators = int(request.form.get("number_of_oscillators", 0))
        spring_lengths = [
            float(request.form.get(f"spring_length_{i}", 10.0))
            for i in range(1, number_of_oscillators + 2)
        ]
        masses = [
            float(request.form.get(f"mass_{i}", 1.0))
            for i in range(1, number_of_oscillators + 1)
        ]
        spring_constants = [
            float(request.form.get(f"spring_constant_{i}", 1.0))
            for i in range(1, number_of_oscillators + 2)
        ]
        damping_coefficient = float(request.form.get("damping", 0.5))
        time_step = float(request.form.get("time_step", 0.005))
        n_states = int(request.form.get("n_states", N_STATES))

        # Create and run simulation
        simulation = OscillatorsSimulation(
            oscillator_masses=masses,
            springs_current_lens=spring_lengths,
            spring_constants=spring_constants,
            damping=damping_coefficient,
            time_step=time_step,
        )
        ani = simulation.create_animation(n_states)
        plots = list(simulation.get_plots().values())[0]

        return render_template(
            "index.html",
            number_of_oscillators=number_of_oscillators,
            masses=masses,
            spring_constants=spring_constants,
            spring_lengths=spring_lengths,
            damping_coefficient=damping_coefficient,
            positions=spring_lengths,  # Assuming the positions are related to spring lengths
            plots=plots,
            time_step=time_step,
            n_states=n_states,
            animation=ani.to_jshtml(),
        )
    else:
        # START PARAMETERS - FIRST EXAMPLE
        number_of_oscillators = 3
        masses = [1.0, 1.5, 2.0]
        spring_constants = [1.0, 1.0, 1.0, 1.0]  # Default spring constants
        spring_lengths = [10, 20, 30, 20]  # Default spring lengths
        damping_coefficient = 0.5
        time_step = 0.005

        # Create and run simulation
        simulation = OscillatorsSimulation(
            oscillator_masses=masses,
            springs_current_lens=spring_lengths,
            spring_constants=spring_constants,
            damping=damping_coefficient,
            time_step=time_step,
        )

        ani = simulation.create_animation(N_STATES)
        plots = list(simulation.get_plots().values())[0]

        return render_template(
            "index.html",
            number_of_oscillators=number_of_oscillators,
            masses=masses,
            spring_constants=spring_constants,
            spring_lengths=spring_lengths,
            damping_coefficient=damping_coefficient,
            positions=spring_lengths,  # Assuming the positions are related to spring lengths
            plots=plots,
            time_step=time_step,
            n_states=N_STATES,
            animation=ani.to_jshtml(),
        )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
