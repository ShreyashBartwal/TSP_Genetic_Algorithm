import random
#from numpy.lib.function_base import append
import tsplib95
import networkx as nx
import math
import sys
from threading import Thread

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


import matplotlib
matplotlib.use('Qt5Agg')
class TSPGeneticAlgorithmGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initial_graph, self.result_graph = nx.Graph(), nx.Graph()
        self.max_length_y, self.is_running, self.file_path = 1000, False, ""
        self.setWindowTitle('TSP Genetic Algorithm')
        self.setMinimumSize(QSize(900, 600))

        # Set a font for the whole application
        font = QFont("Arial", 10)  # You can change "Arial" and size as needed
        self.setFont(font)

        # Main layout
        main_layout = QVBoxLayout()
        self.content_widget = QWidget()

        # Left panel layout for input fields and buttons
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setContentsMargins(6 , 0 , 9, 0)

        # Grid for input fields
        input_grid_layout = QGridLayout()
        input_grid_layout.setSpacing(10)

        self.input_generations = QLineEdit('1000')
        self.input_population_size = QLineEdit('100')
        self.input_mutation_rate = QLineEdit('0.15')
        self.input_elite_percentage = QLineEdit('0.2')
        self.input_random_cities = QLineEdit('30')

        input_grid_layout.addWidget(QLabel('Generations:'), 1, 0)
        input_grid_layout.addWidget(self.input_generations, 1, 1)
        input_grid_layout.addWidget(QLabel('Population size:'), 2, 0)
        input_grid_layout.addWidget(self.input_population_size, 2, 1)
        input_grid_layout.addWidget(QLabel('Mutation rate:'), 3, 0)
        input_grid_layout.addWidget(self.input_mutation_rate, 3, 1)
        input_grid_layout.addWidget(QLabel('Elites (%):'), 4, 0)
        input_grid_layout.addWidget(self.input_elite_percentage, 4, 1)
        input_grid_layout.addWidget(QLabel('Random cities:'), 5, 0)
        input_grid_layout.addWidget(self.input_random_cities, 5, 1)

        left_panel_layout.addLayout(input_grid_layout)

        # Styling buttons
        button_style = """
            QPushButton {
                background-color: gray;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: black;
            }
        """

        self.button_select_file = QPushButton('Select file')
        self.button_select_file.setStyleSheet(button_style)
        self.button_start_random = QPushButton('Start random')
        self.button_start_random.setStyleSheet(button_style)
        self.button_start_from_file = QPushButton('Start from file')
        self.button_start_from_file.setStyleSheet(button_style)

        # Connecting buttons
        self.button_select_file.clicked.connect(self.open_file_dialog)
        self.button_start_random.clicked.connect(self.threaded_random_tsp)
        self.button_start_from_file.clicked.connect(self.threaded_file_tsp)

        # Adding buttons to the layout
        left_panel_layout.addWidget(QLabel('File operations:'))
        left_panel_layout.addWidget(self.button_select_file)
        left_panel_layout.addWidget(self.button_start_random)
        left_panel_layout.addWidget(self.button_start_from_file)

        self.label_running_status = QLabel("")
        self.label_file_path = QLabel("")
        self.label_result_status = QLabel("")
        left_panel_layout.addWidget(self.label_file_path)
        left_panel_layout.addWidget(self.label_running_status)
        left_panel_layout.addWidget(self.label_result_status)

        # Right panel for the graph
        right_panel_layout = QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes_tsp_visualization = self.figure.add_subplot()
        nx.draw(self.result_graph, ax=self.axes_tsp_visualization)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_panel_layout.addWidget(self.canvas)

        # Combine left and right panels
        main_layout.addLayout(left_panel_layout)
        main_layout.addLayout(right_panel_layout)

        # Set layout to content and central widget
        self.content_widget.setLayout(main_layout)
        self.setCentralWidget(self.content_widget)

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "", "", "tsp Files (*.tsp)")
        if file_name:
            self.file_path = file_name
            self.label_file_path.setText(f"File selected: {self.file_path}")

    def threaded_random_tsp(self):
        if not self.is_running:
            self.cities = self.generate_random_points(int(self.input_random_cities.text()))
            Thread(target=self.start_tsp_algorithm, daemon=True).start()

    def threaded_file_tsp(self):
        try:
            problem = tsplib95.load(self.file_path)
            self.cities = problem.node_coords
            if not self.is_running:
                Thread(target=self.start_tsp_algorithm, daemon=True).start()
        except Exception as e:
            print(f"Error loading file: {e}")

    def start_tsp_algorithm(self):
        self.result_graph.clear()
        self.is_running = True
        self.label_running_status.setText("Running...")
        self.tsp_algorithm = GeneticAlgorithm(self)
        self.generations_count = int(self.input_generations.text())
        self.tsp_algorithm.find_optimum(
            len(self.cities),
            self.cities,
            self.generations_count,
            float(self.input_mutation_rate.text()),
            int(self.input_population_size.text()),
            float(self.input_elite_percentage.text())
        )
        if self.tsp_algorithm.generation_list[-1] != self.generations_count:
            self.tsp_algorithm.generation_list.append(self.generations_count)
            self.tsp_algorithm.cost_list.append(self.tsp_algorithm.cost_list[-1])
        self.update_result_graph()
        self.is_running = False
        self.label_running_status.setText("Completed!")

    def update_result_graph(self):
        route_edges = [(self.best.cities_index[i] + 1, self.best.cities_index[i + 1] + 1) for i in range(len(self.best.cities_index) - 1)]
        self.result_graph.add_nodes_from(self.cities.keys())
        self.result_graph.add_edges_from(route_edges)
        self.axes_tsp_visualization.clear()
        nx.draw(self.result_graph, with_labels=True, pos=self.cities, edgelist=route_edges, edge_color='b', ax=self.axes_tsp_visualization, node_size=50)

        self.canvas.draw()
        self.label_result_status.setText(f"Lowest length: {self.tsp_algorithm.best_chromosome.length}")

    def generate_random_points(self, n):
        return {i + 1: (random.uniform(0, 1000), random.uniform(0, 1000)) for i in range(n)}

    

class GeneticAlgorithm:
    def __init__(self, gui=None):
        self.chromosomes = []
        self.total_length = 0
        self.total_fitness = 0
        self.fitness_sum = 0
        self.parent_chromosomes = []
        self.current_generation = 0
        self.stagnation_count = 0
        self.gui = gui
        self.best_chromosome = None
        self.generation_list = []
        self.cost_list = []

    def generate_initial_population(self, size):
        """Generates initial population."""
        for _ in range(size):
            chromosome = Chromosome(self.num_nodes)
            random.shuffle(chromosome.cities_index)
            self.chromosomes.append(chromosome)
        self.calculate_fitness()
        self.normalize_fitness()

    def calculate_fitness(self):
        """Calculates fitness function of all chromosomes in population."""
        self.total_length = 0
        self.fitness_sum = 0
        for chrom in self.chromosomes:
            chrom.age += 1
            if chrom.age >= 50:
                random.shuffle(chrom.cities_index)
                chrom.age = 0
            chrom.calculate_total_length(self.city_coordinates)
            self.total_length += chrom.length
            if chrom.length < self.best_chromosome.length:
                self.best_chromosome.length = chrom.length
                self.best_chromosome.cities_index = list(chrom.cities_index)
                self.best_chromosome.cities_index.append(self.best_chromosome.cities_index[0])
                print("Generation: %d: %.4f" % (self.current_generation, self.best_chromosome.length))
                self.generation_list.append(self.current_generation)
                self.cost_list.append(self.best_chromosome.length)
                if len(self.cost_list) == 1:
                    self.gui.max_y = self.cost_list[0]
                self.gui.update_result_graph()
            self.fitness_sum += 1 / chrom.length

    def normalize_fitness(self):
        """Normalizes fitness score of chromosomes to be in interval <0;1>."""
        for chrom in self.chromosomes:
            chrom.fitness = (1 / chrom.length) / self.fitness_sum
            self.total_fitness += chrom.fitness

    def roulette_wheel_selection(self):
        """Selects a parent chromosome randomly based on fitness."""
        random_selection = random.random()
        cumulative_probability = 0
        for chrom in self.chromosomes:
            cumulative_probability += chrom.fitness
            if random_selection <= cumulative_probability:
                return chrom

    def generate_new_population(self, elite_ratio):
        """Generates new generation of chromosomes. Percentage of elites survives into next generation."""
        self.current_generation += 1
        new_population = []
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=True)
        elite_count = math.floor(elite_ratio * len(self.chromosomes))
        new_population.extend(self.chromosomes[:elite_count])

        for _ in range(len(self.chromosomes) - elite_count):
            parent1 = self.roulette_wheel_selection()
            parent2 = self.roulette_wheel_selection()
            child = self.order_crossover(parent1, parent2)
            self.mutate_random_swap(child)
            new_population.append(child)

        self.chromosomes = new_population
        self.calculate_fitness()
        self.normalize_fitness()

    def order_crossover(self, parent1, parent2):
        """Performs crossover on parent1 and parent2 to create a new child chromosome."""
        start_index = random.randint(0, self.num_nodes - 2)
        end_index = random.randint(start_index + 1, self.num_nodes - 1)

        child = Chromosome(self.num_nodes)
        child.cities_index = [-1 for _ in range(len(parent1.cities_index))]
        j = end_index
        for i in range(start_index, end_index):
            child.cities_index[i] = parent1.cities_index[i]
        for i in range(end_index, len(parent1.cities_index)):
            while True:
                if parent2.cities_index[j] not in child.cities_index:
                    child.cities_index[i] = parent2.cities_index[j]
                    j += 1
                    if j >= len(parent2.cities_index):
                        j = 0
                    break
                j += 1
                if j >= len(parent2.cities_index):
                    j = 0
        for i in range(start_index):
            while True:
                if parent2.cities_index[j] not in child.cities_index:
                    child.cities_index[i] = parent2.cities_index[j]
                    break
                j += 1
        return child

    def mutate_random_swap(self, chrom):
        """Performs mutation by swapping random genes."""
        if random.random() <= self.mutation_probability:
            first_index = random.randint(0, self.num_nodes - 1)
            second_index = random.randint(0, self.num_nodes - 1)
            chrom.cities_index[first_index], chrom.cities_index[second_index] = chrom.cities_index[second_index], chrom.cities_index[first_index]

    def mutate_swap_neighbors(self, chrom):
        """Performs mutation by swapping adjacent genes."""
        if random.random() <= self.mutation_probability:
            first_index = random.randint(0, self.num_nodes - 1)
            second_index = first_index + 1
            if second_index > len(chrom.cities_index) - 1:
                second_index = first_index - 1
            chrom.cities_index[first_index], chrom.cities_index[second_index] = chrom.cities_index[second_index], chrom.cities_index[first_index]

    def find_optimum(self, num_nodes, city_coordinates, generations, mutation_probability, initial_population_size, elite_percentage):
        """For a number of generations, the algorithm tries to find the solution of TSP for given cities."""
        self.num_nodes = num_nodes
        self.best_chromosome = Chromosome(self.num_nodes)
        self.best_chromosome.length = sys.maxsize
        self.gui.best = self.best_chromosome
        self.city_coordinates = city_coordinates
        self.mutation_probability = mutation_probability
        self.generate_initial_population(initial_population_size)
        for _ in range(generations):
            self.stagnation_count += 1
            self.generate_new_population(elite_percentage)

class Chromosome:
    def __init__(self, num_cities):
        self.cities_index = list(range(0, num_cities))
        self.length = 0
        self.fitness = 0
        self.age = 0

    def length_between_two(self, start, end):
        """Returns length between two points."""
        return math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

    def calculate_total_length(self, cities):
        """Computes length of the whole route."""
        self.length = 0
        for i in range(len(cities) - 1):
            start_index = self.cities_index[i]
            end_index = self.cities_index[i + 1]
            self.length += self.length_between_two(cities[start_index + 1], cities[end_index + 1])
        self.length += self.length_between_two(cities[len(cities)], cities[1])


def main():
    app = QApplication(sys.argv)
    window = TSPGeneticAlgorithmGUI()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()