from flask import Flask, render_template
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from io import BytesIO
import base64

from simulation import OscillatorsSimulation

app = Flask(__name__)

# Mockup data for demonstration
def generate_plot():
    x = list(range(100))
    
    # Generate three sets of mock data
    y1 = [random.uniform(-1, 1) for _ in x]
    y2 = [random.uniform(-1, 1) for _ in x]
    y3 = [random.uniform(-1, 1) for _ in x]

    plt.plot(x, y1, label='Function 1')
    plt.plot(x, y2, label='Function 2')
    plt.plot(x, y3, label='Function 3')
    
    plt.title("Mockup Data")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    # Save plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image to base64 for embedding in HTML
    return base64.b64encode(img.getvalue()).decode('utf-8')


# Flask route
@app.route('/')
def index():
    # Mockup data for the template
    number_of_oscillators = 3
    masses = [1.0, 1.5, 2.0]
    damping_coefficient = 0.5
    elastic_collisions = True
    positions = [0.0, 0.5, 1.0]

    # Generate mockup plots
    oscillator_phases_plot = generate_plot()
    forces_plot = generate_plot()
    velocities_plot = generate_plot()
    positions_plot = generate_plot()

    # Create Matplotlib Animation
    simulation = OscillatorsSimulation([2,2])
    ani = simulation.create_animation(10)

    return render_template('index.html', 
                           number_of_oscillators=number_of_oscillators,
                           masses=masses,
                           damping_coefficient=damping_coefficient,
                           elastic_collisions=elastic_collisions,
                           positions=positions,
                           oscillator_phases_plot=oscillator_phases_plot,
                           forces_plot=forces_plot,
                           velocities_plot=velocities_plot,
                           positions_plot=positions_plot,
                           animation=ani.to_jshtml())

if __name__ == '__main__':
    app.run(debug=True)
