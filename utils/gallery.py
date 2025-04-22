from PyQt6.QtWidgets import QMainWindow, QFileDialog, QApplication, QLabel, QScrollArea, QToolButton, QWidget, QVBoxLayout, QGridLayout, QFormLayout
from PyQt6.QtGui import QPixmap, QImage, QColor, QPalette
from PyQt6.QtCore import Qt, QPoint
import sys, cv2, os
import yaml

colors = [
    QColor('black'), QColor('darkgrey'), QColor('red'), QColor('green'),
    QColor('blue'), QColor('yellow'), QColor('cyan'), QColor('magenta'),
    QColor('orange'), QColor('purple'), QColor('brown'), QColor('gray'),
    QColor('darkred'), QColor('darkgreen'), QColor('darkblue'), QColor('darkcyan'),
    QColor('darkmagenta'), QColor('darkorange'), QColor('darkviolet'), QColor('darkgray'),
    QColor('white'), QColor('lightgray'), QColor('lightgrey'), QColor('silver'),
    QColor('gold'), QColor('olive'), QColor('teal'), QColor('navy'),
    QColor('maroon'), QColor('lime'), QColor('fuchsia'), QColor('aqua'),
    QColor('forestgreen'), QColor('midnightblue'), QColor('dodgerblue'), QColor('firebrick'),
    QColor('chocolate'), QColor('saddlebrown'), QColor('sandybrown'), QColor('indianred'),
    QColor('rosybrown'), QColor('sienna'), QColor('peru'), QColor('goldenrod'),
    QColor('darkgoldenrod'), QColor('palevioletred'), QColor('mediumvioletred'), QColor('mediumpurple'),
    QColor('rebeccapurple'), QColor('mediumorchid'), QColor('darkorchid'), QColor('darkslateblue'),
    QColor('mediumslateblue'), QColor('slateblue'), QColor('darkslategray'), QColor('darkslategrey'),
    QColor('lightslategray'), QColor('lightslategrey'), QColor('lightsteelblue'), QColor('lightseagreen'),
    QColor('mediumseagreen'), QColor('darkseagreen'), QColor('seagreen'), QColor('springgreen'),
    QColor('yellowgreen'), QColor('lawngreen'), QColor('chartreuse'), QColor('greenyellow'),
    QColor('lightgreen'), QColor('palegreen'), QColor('honeydew'), QColor('mintcream'),
    QColor('azure'), QColor('aliceblue'), QColor('ghostwhite'), QColor('whitesmoke'),
    QColor('seashell'), QColor('beige'), QColor('oldlace'), QColor('floralwhite'),
    QColor('antiquewhite'), QColor('linen'), QColor('papayawhip'), QColor('blanchedalmond'),
    QColor('bisque'), QColor('moccasin'), QColor('wheat'), QColor('burlywood'),
    QColor('tan'), QColor('khaki'), QColor('darkkhaki'), QColor('cornsilk'),
    QColor('lemonchiffon'), QColor('lightgoldenrodyellow'), QColor('lightyellow'), QColor('ivory'),
    QColor('lightcyan'), QColor('aliceblue'), QColor('lightskyblue'), QColor('powderblue'),
    QColor('lightblue'), QColor('skyblue'), QColor('deepskyblue'), QColor('steelblue'),
    QColor('cadetblue'), QColor('darkturquoise'), QColor('mediumturquoise'), QColor('turquoise'),
    QColor('lightseagreen'), QColor('darkcyan'), QColor('darkslategray'), QColor('darkslategrey'),
    QColor('lightcyan'), QColor('paleturquoise'), QColor('aqua'), QColor('aquamarine'),
    QColor('mediumaquamarine'), QColor('mediumspringgreen'), QColor('springgreen'), QColor('mintcream'),
    QColor('honeydew'), QColor('lightgreen'), QColor('palegreen'), QColor('darkseagreen'),
    QColor('mediumseagreen'), QColor('seagreen'), QColor('forestgreen'), QColor('green'),
    QColor('darkgreen'), QColor('greenyellow'), QColor('chartreuse'), QColor('lawngreen'),
    QColor('lime'), QColor('limegreen'), QColor('yellowgreen'), QColor('darkolivegreen'),
    QColor('olivedrab'), QColor('olive'), QColor('beige'), QColor('lightgoldenrodyellow'),
    QColor('lightyellow'), QColor('yellow'), QColor('gold'), QColor('darkgoldenrod'),
    QColor('goldenrod'), QColor('palegoldenrod'), QColor('darkkhaki'), QColor('khaki'),
    QColor('lightyellow'), QColor('ivory'), QColor('cornsilk'), QColor('lemonchiffon'),
    QColor('lightgoldenrodyellow'), QColor('lightyellow'), QColor('lightcyan'), QColor('aliceblue'),
    QColor('azure'), QColor('mintcream'), QColor('honeydew'), QColor('lightgreen'),
    QColor('palegreen'), QColor('darkseagreen'), QColor('mediumseagreen'), QColor('seagreen'),
    QColor('forestgreen'), QColor('green'), QColor('darkgreen'), QColor('greenyellow'),
    QColor('chartreuse'), QColor('lawngreen'), QColor('lime'), QColor('limegreen'),
    QColor('yellowgreen'), QColor('darkolivegreen'), QColor('olivedrab'), QColor('olive'),
    QColor('beige'), QColor('lightgoldenrodyellow'), QColor('lightyellow'), QColor('yellow'),
    QColor('gold'), QColor('darkgoldenrod'), QColor('goldenrod'), QColor('palegoldenrod'),
    QColor('darkkhaki'), QColor('khaki'), QColor('lightyellow'), QColor('ivory'),
    QColor('cornsilk'), QColor('lemonchiffon'), QColor('lightgoldenrodyellow'), QColor('lightyellow'),
    QColor('lightcyan'), QColor('aliceblue'), QColor('azure'), QColor('mintcream'),
    QColor('honeydew'), QColor('lightgreen'), QColor('palegreen'), QColor('darkseagreen'),
    QColor('mediumseagreen'), QColor('seagreen'), QColor('forestgreen'), QColor('green'),
    QColor('darkgreen'), QColor('greenyellow'), QColor('chartreuse'), QColor('lawngreen'),
    QColor('lime'), QColor('limegreen'), QColor('yellowgreen'), QColor('darkolivegreen'),
    QColor('olivedrab'), QColor('olive'), QColor('beige'), QColor('lightgoldenrodyellow'),
    QColor('lightyellow'), QColor('yellow'), QColor('gold'), QColor('darkgoldenrod'),
    QColor('goldenrod'), QColor('palegoldenrod'), QColor('darkkhaki'), QColor('khaki')
]

def gridInit(dir):
    # Iterate over subdirectories to load images and create buttons/labels
    grid_layout = QGridLayout()
    base_dir = dir
    row, col = 0, 0
    for i, subdir in enumerate(os.listdir(base_dir)):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            for img_file in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, img_file)
                if os.path.isfile(img_path) and img_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                    # Create label to display image
                    image_label = QLabel()
                    pixmap = QPixmap(img_path)
                    image_label.setPixmap(pixmap.scaledToWidth(200))  # Scale image width to 200

                    # Create tool button with the color and class name
                    tool_button = QToolButton()
                    color_name = colors[i % len(colors)].name()
                    tool_button.setStyleSheet(f"background-color: {color_name};")
                    tool_button.setAutoFillBackground(True)
                    tool_button.update()

                    # Create a container widget to hold the tool button and image
                    container_widget = QWidget()
                    container_layout = QVBoxLayout()
                    container_widget.setLayout(container_layout)
                    container_layout.addWidget(tool_button, alignment=Qt.AlignmentFlag.AlignTop)
                    container_layout.addWidget(image_label)

                    # Add the container widget to the grid layout
                    grid_layout.addWidget(container_widget, row, col)
                    col += 1
                    if col >= 4:  # 3 images per row
                        col = 0
                        row += 1
    return grid_layout

def yoloInit_(dir):
    grid_layout = QGridLayout()
    base_dir = dir
    row, col = 0, 0
    for img_file in os.listdir(dir):
        img_path = os.path.join(dir, img_file)
        if os.path.isfile(img_path) and img_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            # Create a container widget to hold the tool button and image
            container_widget = QWidget()
            container_layout = QVBoxLayout()
            container_widget.setLayout(container_layout)

            # Add the container widget to the grid layout
            grid_layout.addWidget(container_widget, row, col)
            col += 1
            if col >= 4:  # 3 images per row
                col = 0
                row += 1
    return grid_layout

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def get_image_classes(label_path, class_names):
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        classes = set(int(line.split()[0]) for line in lines)
    return [class_names[cls] for cls in classes]

def yoloInit(base_dir):
    # Load the data.yaml file
    data_yaml_path = os.path.join(base_dir, 'data.yaml')
    data_yaml = load_yaml(data_yaml_path)
    class_names = data_yaml['names']
    num_classes = data_yaml['nc']
    
    # Define colors for the classes (this can be customized)
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']  # Add more colors if needed

    # Create grid layout
    grid_layout = QGridLayout()
    row, col = 0, 0

    # Define subdirectories for train, val, and test
    subdirs = ['train', 'valid']
    for subdir in subdirs:
        image_dir = os.path.join(base_dir, subdir, 'images')
        label_dir = os.path.join(base_dir, subdir, 'labels')
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            continue
        
        for img_file in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_file)
            if os.path.isfile(img_path) and img_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(label_dir, label_file)
                
                # Get the classes present in the image
                classes_in_image = get_image_classes(label_path, class_names)
                
                # Create label to display image
                image_label = QLabel()
                pixmap = QPixmap(img_path)
                image_label.setPixmap(pixmap.scaledToWidth(200))  # Scale image width to 200

                # Create a container widget to hold the tool buttons and image
                container_widget = QWidget()
                container_layout = QVBoxLayout()
                container_widget.setLayout(container_layout)
                
                # Add tool buttons for each class in the image
                for class_name in classes_in_image:
                    tool_button = QToolButton()
                    color = colors[class_names.index(class_name) % len(colors)]
                    tool_button.setText(class_name)
                    tool_button.setStyleSheet(f"background-color: {color};")
                    tool_button.setAutoFillBackground(True)
                    container_layout.addWidget(tool_button, alignment=Qt.AlignmentFlag.AlignTop)
                
                container_layout.addWidget(image_label)
                
                # Add the container widget to the grid layout
                grid_layout.addWidget(container_widget, row, col)
                col += 1
                if col >= 4:  # 4 images per row
                    col = 0
                    row += 1
    
    return grid_layout

def legendsInit(dir):
    form_layout = QFormLayout()
    classes_name = []
    widgets = {}

    for subdir in os.listdir(dir):
        classes_name.append(subdir)

    for i, class_name in enumerate(classes_name):
        tool_button = QToolButton()
        palette = tool_button.palette()
        palette.setColor(QPalette.ColorRole.Button, colors[i % len(colors)])
        tool_button.setAutoFillBackground(True)
        tool_button.setPalette(palette)
        tool_button.update()

        label = QLabel(class_name)

        form_layout.addRow(tool_button, label)

    return form_layout

class LegendsManager:
    def __init__(self, dir, colors):
        self.form_layout = QFormLayout()
        self.classes_name = []
        self.widgets = {}  # Dictionary to store references to the buttons and labels
        
        for subdir in os.listdir(dir):
            self.classes_name.append(subdir)
        
        for i, class_name in enumerate(self.classes_name):
            tool_button = QToolButton()
            palette = tool_button.palette()
            palette.setColor(QPalette.ColorRole.Button, colors[i % len(colors)])
            tool_button.setAutoFillBackground(True)
            tool_button.setPalette(palette)
            tool_button.update()
            
            label = QLabel(class_name)
            
            # Store the button and label reference
            self.widgets[class_name] = {'button': tool_button, 'label': label}
            
            # Add the tool button and label to the form layout
            self.form_layout.addRow(tool_button, label)

    def get_layout(self):
        return self.form_layout
    
    def modify_button(self, class_name, new_color):
        if class_name in self.widgets:
            button = self.widgets[class_name]['button']
            palette = button.palette()
            palette.setColor(QPalette.ColorRole.Button, new_color)
            button.setPalette(palette)
            button.update()
        else:
            print("Class name not found")
    
    def highlight_label(self, class_name, default_color=QColor(0, 0, 0)):
        # Reset all labels to the default color
        for widget in self.widgets.values():
            label = widget['label']
            palette = label.palette()
            palette.setColor(QPalette.ColorRole.WindowText, default_color)
            label.setPalette(palette)
            label.update()
        
        # Highlight the specified label
        if class_name in self.widgets:
            label = self.widgets[class_name]['label']
            palette = label.palette()
            palette.setColor(QPalette.ColorRole.WindowText, colors[0])
            label.setPalette(palette)
            label.update()
        else:
            print("Class name not found")

def create_legends_instance(name, dir, colors, legends_instances):
    legends_manager = LegendsManager(dir, colors)
    legends_instances[name] = legends_manager
    return legends_manager