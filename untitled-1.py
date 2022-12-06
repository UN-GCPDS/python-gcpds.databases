# FigureStream replace any Figure object
from figurestream import FigureStream

import numpy as np
from datetime import datetime

# FigureStream can be used like any Figure object
stream = FigureStream()
sub = stream.add_subplot(111)
x = np.linspace(0, 3, 1000)

# Update animation loop
while True:
    sub.clear()  # clear the canvas

    # ------------------------------------------------------------------------
    # Any plot operation
    sub.set_title('FigureStream')
    sub.set_xlabel('Time [s]')
    sub.set_ylabel('Amplitude')
    sub.plot(x, np.sin(2 * np.pi * 2 * (x + datetime.now().timestamp())))
    sub.plot(x, np.sin(2 * np.pi * 0.5 * (x + datetime.now().timestamp())))
    # ------------------------------------------------------------------------

    stream.feed()  # push the frame into the server
