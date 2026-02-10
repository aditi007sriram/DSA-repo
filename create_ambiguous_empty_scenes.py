# create_ambiguous_empty_scenes.py
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os

print("Creating ambiguous empty scenes (more likely to trigger hallucinations)...")
os.makedirs('ambiguous_empty', exist_ok=True)

def add_noise(img):
    arr = np.array(img)
    noise = np.random.normal(0, 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def create_foggy_road():
    """Foggy road - fog creates ambiguous shapes"""
    img = Image.new('RGB', (640, 480))
    draw = ImageDraw.Draw(img)
    
    # Very foggy sky
    for y in range(200):
        gray = int(200 - y * 0.2)
        draw.rectangle([0, y, 640, y+1], fill=(gray, gray, gray+10))
    
    # Barely visible road
    draw.polygon([(0, 200), (280, 480), (360, 480), (640, 200)], 
                 fill=(90, 90, 95))
    
    # Faint lane markings (could look like objects)
    for i in range(6):
        y = 250 + i * 40
        w = 10 + i * 2
        draw.rectangle([320-w, y, 320+w, y+15], fill=(180, 180, 180))
    
    # Add heavy blur (creates ambiguous blobs)
    img = img.filter(ImageFilter.GaussianBlur(3))
    img = add_noise(img)
    return img

def create_dark_parking_lot():
    """Dark parking lot with shadows"""
    img = Image.new('RGB', (640, 480))
    draw = ImageDraw.Draw(img)
    
    # Dark sky
    draw.rectangle([0, 0, 640, 120], fill=(30, 30, 40))
    
    # Very dark asphalt
    draw.rectangle([0, 120, 640, 480], fill=(25, 25, 30))
    
    # Faint white lines (could be misinterpreted)
    for i in range(8):
        x = 50 + i * 80
        draw.rectangle([x, 200, x+4, 480], fill=(80, 80, 85))
    
    # Shadow patches (ambiguous dark regions)
    draw.ellipse([100, 300, 250, 400], fill=(15, 15, 20))
    draw.ellipse([400, 250, 550, 380], fill=(18, 18, 23))
    
    img = add_noise(img)
    return img

def create_blurry_indoor():
    """Blurry indoor space with ambiguous shapes"""
    img = Image.new('RGB', (640, 480))
    draw = ImageDraw.Draw(img)
    
    # Wall
    draw.rectangle([0, 0, 640, 300], fill=(200, 195, 190))
    
    # Floor
    draw.polygon([(0, 300), (640, 300), (550, 480), (90, 480)],
                 fill=(160, 155, 150))
    
    # Create ambiguous rectangular shapes (not objects, just floor patterns)
    # These might trigger "person" or "chair" detections
    draw.rectangle([150, 350, 200, 450], fill=(140, 135, 130))
    draw.rectangle([450, 320, 490, 420], fill=(145, 140, 135))
    
    # Heavy blur
    img = img.filter(ImageFilter.GaussianBlur(4))
    img = add_noise(img)
    return img

def create_highway_with_shadows():
    """Highway with shadow patches that might look like vehicles"""
    img = Image.new('RGB', (640, 480))
    draw = ImageDraw.Draw(img)
    
    # Sky
    for y in range(180):
        blue = int(180 - y * 0.3)
        draw.rectangle([0, y, 640, y+1], fill=(120, 140, blue))
    
    # Road
    draw.polygon([(0, 180), (270, 480), (370, 480), (640, 180)],
                 fill=(70, 70, 75))
    
    # Shadow patches (dark rectangles that could look like cars)
    # Bottom center (where cars usually are)
    draw.rectangle([200, 380, 270, 450], fill=(35, 35, 40))
    draw.rectangle([380, 360, 440, 430], fill=(40, 40, 45))
    
    # Lane lines
    for i in range(7):
        y = 220 + i * 40
        w = 6 + i * 2
        draw.rectangle([320-w, y, 320+w, y+12], fill=(220, 220, 180))
    
    img = img.filter(ImageFilter.GaussianBlur(1.5))
    img = add_noise(img)
    return img

def create_street_with_light_spots():
    """Street with bright spots that could be misinterpreted"""
    img = Image.new('RGB', (640, 480))
    draw = ImageDraw.Draw(img)
    
    # Sky
    draw.rectangle([0, 0, 640, 100], fill=(150, 170, 190))
    
    # Buildings (sides)
    draw.rectangle([0, 100, 120, 480], fill=(100, 95, 90))
    draw.rectangle([520, 100, 640, 480], fill=(95, 90, 85))
    
    # Street
    draw.polygon([(120, 100), (520, 100), (450, 480), (190, 480)],
                 fill=(65, 65, 70))
    
    # Bright reflective spots (could trigger detections)
    # Center-bottom (where people/cars usually are)
    draw.ellipse([250, 350, 300, 400], fill=(180, 180, 185))
    draw.ellipse([350, 320, 390, 370], fill=(175, 175, 180))
    draw.ellipse([200, 300, 230, 340], fill=(170, 170, 175))
    
    img = img.filter(ImageFilter.GaussianBlur(2))
    img = add_noise(img)
    return img

def create_textured_wall():
    """Wall with texture patterns that might look like objects"""
    img = Image.new('RGB', (640, 480))
    draw = ImageDraw.Draw(img)
    
    # Base wall
    draw.rectangle([0, 0, 640, 480], fill=(200, 195, 190))
    
    # Create vertical streaks (could look like people standing)
    for x in range(0, 640, 100):
        draw.rectangle([x+30, 150, x+60, 450], fill=(180, 175, 170))
    
    # Circular stains (ambiguous blobs)
    draw.ellipse([100, 200, 180, 280], fill=(190, 185, 180))
    draw.ellipse([450, 250, 520, 320], fill=(185, 180, 175))
    
    img = img.filter(ImageFilter.GaussianBlur(2.5))
    img = add_noise(img)
    return img

# Create multiple variations
scenes = [
    ('foggy_road_1.jpg', create_foggy_road()),
    ('foggy_road_2.jpg', create_foggy_road()),
    ('foggy_road_3.jpg', create_foggy_road()),
    ('dark_parking_1.jpg', create_dark_parking_lot()),
    ('dark_parking_2.jpg', create_dark_parking_lot()),
    ('dark_parking_3.jpg', create_dark_parking_lot()),
    ('blurry_indoor_1.jpg', create_blurry_indoor()),
    ('blurry_indoor_2.jpg', create_blurry_indoor()),
    ('highway_shadows_1.jpg', create_highway_with_shadows()),
    ('highway_shadows_2.jpg', create_highway_with_shadows()),
    ('highway_shadows_3.jpg', create_highway_with_shadows()),
    ('street_lights_1.jpg', create_street_with_light_spots()),
    ('street_lights_2.jpg', create_street_with_light_spots()),
    ('textured_wall_1.jpg', create_textured_wall()),
    ('textured_wall_2.jpg', create_textured_wall()),
]

for filename, img in scenes:
    filepath = os.path.join('ambiguous_empty', filename)
    img.save(filepath, quality=95)
    print(f"✓ Created: {filename}")

print(f"\n✓ Created {len(scenes)} ambiguous empty scenes")
print("These have patterns that look like objects but aren't!")