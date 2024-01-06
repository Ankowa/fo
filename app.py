from flask import Flask, render_template
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from simulation import OscillatorsSimulation

app = Flask(__name__)

# Flask route
@app.route('/')
def index():
    # Mockup data for the template
    number_of_oscillators = 3
    masses = [1.0, 1.5, 2.0]
    damping_coefficient = 0.5
    elastic_collisions = True
    positions = [0.0, 0.5, 1.0]


    # Create Matplotlib Animation
    simulation = OscillatorsSimulation([2,2, 1], [10, 40, 10, 20])
    ani = simulation.create_animation(2000)

    plots = simulation.get_plots()
    oscillators_plot = list(plots.values())[0]
    
    # Create Matplotlib Animation
    simulation = OscillatorsSimulation([2,2], [10, 40, 10])
    ani = simulation.create_animation(1000)

    return render_template('index.html', 
                           number_of_oscillators=number_of_oscillators,
                           masses=masses,
                           damping_coefficient=damping_coefficient,
                           elastic_collisions=elastic_collisions,
                           positions=positions,
                           oscillators_plot=oscillators_plot,
                           animation=ani.to_jshtml())

if __name__ == '__main__':
    app.run(debug=True)
