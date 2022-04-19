import threading

import PySimpleGUI as sg
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import logging

cpk_colours = {
                'O': 'red',
                'N': 'blue',
                'H': 'white',
                'Cl': 'green',
                'S': 'yellow',
                'C': 'black',
                'F': 'grey',
                'K': '#08F0F3',
                'Cu': 'orange',
                'Br': '#BFF308',
                'Li': '#8B3E02',
                'Ca': '#B104C2',
                'Mg': '#568903',
                'Si': '#F8F895',
                'I': '#70BEBF',
                'P': 'pink'
            }

logging.basicConfig(filename='server.log',
                    format='%(asctime)s %(message)s',
                    encoding='utf-8', level=logging.INFO)

class NarupaRunner:

    def __init__(self):
        from narupa.app import NarupaFrameApplication
        self.frame_server = NarupaFrameApplication.basic_server(port=0)
        self.drawing_thread = None
        self.pill2kill = None

    def draw(self, molecules_paths):
        from narupa.mdanalysis import mdanalysis_to_frame_data
        import MDAnalysis as mda

        universe = mda.Universe(molecules_paths[0])

        # atoms = reduce(lambda l, r: attach(l, r, 10.), map(read, molecules_paths))
        frame = mdanalysis_to_frame_data(universe)

        logging.info("Draw file: " + molecules_paths[0])
        logging.info("Initial frame: " + str(frame))
        if self.drawing_thread is not None:
            self.pill2kill.set()

        self.frame_server.frame_publisher.send_frame(0, frame)

        def move(stop_event):
            while not stop_event.wait(0.05):
                noise = np.random.normal(0, 1, len(frame.arrays['particle.positions'])) / 500.
                frame.arrays['particle.positions'] += noise
                self.frame_server.frame_publisher.send_frame(0, frame)

        self.pill2kill = threading.Event()
        self.drawing_thread = threading.Thread(target=move, args=(self.pill2kill,))
        self.drawing_thread.start()
        self.design()

    def design(self):
        from narupa.app import NarupaImdClient
        client = NarupaImdClient.connect_to_single_server(port=self.frame_server.port)

        client.subscribe_multiplayer()

        with client.root_selection.modify() as selection:
            selection.interaction_method = 'group'

            selection.renderer = {
                'color': {
                    'type': 'cpk',
                    'scheme': cpk_colours,
                }
            }

    def close(self):
        logging.info("Closing app")
        self.frame_server.close()
        if self.drawing_thread is not None:
            self.pill2kill.set()


file_list_column = [
    [
        sg.Text("Files"),
        sg.In(size=(25, 1), enable_events=True, key="-FILES-"),
        sg.FilesBrowse(),
    ],
    [
        sg.Listbox(values=[], enable_events=True, size=(80, 20), key="-FILE LIST-", auto_size_text=True)
    ],
]

file_runner_column = [
    [sg.Text("Choose a file from the list:")],
    [sg.Text(size=(100, 1), key="-TOUT-")],
    [sg.Button('Run', key="-RUN-")],
    [sg.Text("Current running:")],
    [sg.Text(size=(100, 1), key="-CURRENT_RUNNING-")],
    [sg.Image(key="-VIEW-")],
    [sg.Canvas(key="-CANVAS-")],
]

layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(file_runner_column),
    ]
]

window = sg.Window("Narupa Runner", layout, resizable=True)
nr = NarupaRunner()

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "OK" or event == "Exit" or event == sg.WIN_CLOSED:
        nr.close()
        break
    if event == "-FILES-":
        files = values["-FILES-"]
        window["-FILE LIST-"].update(files.split(';'))
    if event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = values["-FILE LIST-"][0]
            window["-TOUT-"].update(filename)
        except:
            pass
    if event == "-RUN-":  # A file was chosen from the listbox
        try:
            r = nr.draw([values["-FILE LIST-"][0]])
            window["-CURRENT_RUNNING-"].update(values["-FILE LIST-"][0])
        except Exception as e:
            window["-CURRENT_RUNNING-"].update("Error: " + str(e))
            print(e)
            pass
window.close()
