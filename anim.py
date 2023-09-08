import drawsvg as draw

d = draw.Drawing(
    400,
    200,
    origin="center",
    animation_config=draw.types.SyncedAnimationConfig(
        # Animation configuration
        duration=8,  # Seconds
        show_playback_progress=True,
        repeat_count="indefinite",
    ),
)
d.append(draw.Rectangle(-200, -100, 400, 200, fill="#eee"))  # Background
d.append(draw.Circle(0, 0, 40, fill="green"))  # Center circle

# Animation
circle = draw.Circle(0, 0, 0, fill="gray")  # Moving circle
circle.add_key_frame(0, cx=-100, cy=0, r=0)
circle.add_key_frame(2, cx=0, cy=-100, r=40)
circle.add_key_frame(4, cx=100, cy=0, r=0)
circle.add_key_frame(6, cx=0, cy=100, r=40)
circle.add_key_frame(8, cx=-100, cy=0, r=0)
d.append(circle)
r = draw.Rectangle(0, 0, 0, 0, fill="silver")  # Moving square
r.add_key_frame(0, x=-100, y=0, width=0, height=0)
r.add_key_frame(2, x=0 - 20, y=-100 - 20, width=40, height=40)
r.add_key_frame(4, x=100, y=0, width=0, height=0)
r.add_key_frame(6, x=0 - 20, y=100 - 20, width=40, height=40)
r.add_key_frame(8, x=-100, y=0, width=0, height=0)
d.append(r)

# Changing text
draw.native_animation.animate_text_sequence(
    d, [0, 2, 4, 6], ["0", "1", "2", "3"], 30, 0, 1, fill="yellow", center=True
)

# Save as a standalone animated SVG or HTML
d.save_svg("playback-controls.svg")

# Display in Jupyter notebook
# d.display_image()  # Display SVG as an image (will not be interactive)
# d.display_iframe()  # Display as interactive SVG (alternative)
# d.as_gif('orbit.gif', fps=10)  # Render as a GIF image, optionally save to file
# d.as_mp4('orbig.mp4', fps=60, verbose=True)  # Render as an MP4 video, optionally save to file
# d.as_spritesheet('orbit-spritesheet.png', row_length=10, fps=3)  # Render as a spritesheet
d.display_inline()  # Display as interactive SVG
