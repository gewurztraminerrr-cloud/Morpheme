from PIL import Image
import sys

def process_image(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    
    data = img.getdata()
    width, height = img.size
    
    min_x, min_y = width, height
    max_x, max_y = 0, 0
    
    for y in range(height):
        for x in range(width):
            pixel = data[y * width + x]
            # Ignore transparent or very white pixels to find the core logo bounds
            if pixel[3] > 10 and not (pixel[0] > 245 and pixel[1] > 245 and pixel[2] > 245):
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y
                
    if min_x < max_x and min_y < max_y:
        logo = img.crop((min_x, min_y, max_x, max_y))
    else:
        logo = img
        
    target_size = 1024
    
    # Scale up the logo to fill 85% of the icon (making it much larger!)
    scale_factor = (target_size * 0.85) / max(logo.size[0], logo.size[1])
    new_w = int(logo.size[0] * scale_factor)
    new_h = int(logo.size[1] * scale_factor)
    logo_resized = logo.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Use Morpheme's dark background color (#0D1117)
    bg_color = (13, 17, 23, 255)
    
    final_img = Image.new("RGBA", (target_size, target_size), bg_color)
    
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    final_img.paste(logo_resized, (offset_x, offset_y), logo_resized)
    
    # Convert to RGB to remove alpha channel (required by Apple App Store)
    final_img_rgb = final_img.convert("RGB")
    final_img_rgb.save(output_path)

if __name__ == "__main__":
    process_image(sys.argv[1], sys.argv[2])
