from flask import Flask, render_template
import random
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Mockup data for demonstration
def generate_plot():
    x = range(100)
    y = [random.uniform(-1, 1) for _ in x]

    plt.plot(x, y)
    plt.title("Mockup Data")
    plt.xlabel("Time")
    plt.ylabel("Value")

    # Save plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode the image to base64 for embedding in HTML
    return base64.b64encode(img.getvalue()).decode('utf-8')

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

    return render_template('index.html', 
                           number_of_oscillators=number_of_oscillators,
                           masses=masses,
                           damping_coefficient=damping_coefficient,
                           elastic_collisions=elastic_collisions,
                           positions=positions,
                           oscillator_phases_plot=oscillator_phases_plot,
                           forces_plot=forces_plot,
                           velocities_plot=velocities_plot,
                           positions_plot=positions_plot)

if __name__ == '__main__':
    app.run(debug=True)
