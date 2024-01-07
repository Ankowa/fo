import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_color(f, fmax, fmin):
    if f > 0:
        return (1 - f / fmax, 1 - 0.5 * (f / fmax), 0)
    else:
        return (1, 1 - f / fmin, 0)


def create_ani():
    l = 10
    k = 1
    d1 = 3
    d2 = 5
    d3 = l - d1 - d2

    m1 = 3
    m2 = 6

    v10 = 0
    v20 = 0
    x10 = 2
    x20 = 9
    dt = 0.0001
    x1i, x2i, v1i, v2i, f11i, f12i, f21i, f22i, ti = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        np.arange(0, 5, dt),
    )

    x1, x2 = x10, x20
    v1, v2 = v10, v20
    for t in ti:
        dd1 = x1 - d1
        dd2 = x2 - x1 - d2
        dd3 = l - x2 - d3

        f11 = -k * dd1
        f12 = -0.5 * k * dd2
        fw1 = f11 - f12

        f21 = -0.5 * k * dd2
        f22 = -k * dd3
        fw2 = f21 - f22

        a1 = fw1 / m1
        v1 += a1 * dt
        x1 += v1 * dt

        a2 = fw2 / m2
        v2 += a2 * dt
        x2 += v2 * dt

        x1i.append(x1)
        x2i.append(x2)
        v1i.append(v1)
        v2i.append(v2)
        f11i.append(f11)
        f12i.append(f12)
        f21i.append(f21)
        f22i.append(f22)

    fig, ax = plt.subplots()
    fmax = max(max(f11i), max(f12i), max(f21i), max(f22i))
    fmin = min(min(f11i), min(f12i), min(f21i), min(f22i))

    def animate(i):
        ax.clear()
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, 10)
        ax.scatter(x1i[i], 0, c="b")
        ax.scatter(x2i[i], 0, c="m")
        ax.plot([0, x1i[i]], [0, 0], c=get_color(f11i[i], fmax, fmin))
        ax.plot(
            [x1i[i], (x2i[i] + x1i[i]) / 2], [0, 0], c=get_color(f12i[i], fmax, fmin)
        )
        ax.plot(
            [(x2i[i] + x1i[i]) / 2, x2i[i]], [0, 0], c=get_color(f22i[i], fmax, fmin)
        )
        ax.plot([x2i[i], l], [0, 0], c=get_color(f21i[i], fmax, fmin))

    ani = animation.FuncAnimation(fig, animate, frames=range(0, len(ti), len(ti) // 10))
    plt.close()
    return ani
