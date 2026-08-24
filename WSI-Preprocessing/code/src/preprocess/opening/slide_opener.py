from openslide import OpenSlide

class SlideReader:
    def __init__(self, slide_path):
        self.slide = OpenSlide(slide_path)
    
    def get_info(self):
        return {
            "levels": self.slide.level_count,
            "dimensions": self.slide.level_dimensions,
        }