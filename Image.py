class Image:
    def __init__(self, width, height,grid=7):
        self.width = width
        self.height = height
        self.grid_width = width // grid
        self.grid_height = height // grid
