# mikltea sprite

Created with the built-in image-generation tool. Crazy Turtle's existing `avatar.png` was inspected as the resolution/style reference: 30 × 30 RGBA pixels. The generated source is retained in `bubble_tea_source.png`; the installed game asset is `../avatar.png`. `avatar_preview.png` is a 10× nearest-neighbor preview of the actual game sprite.

The game already discovers `agent_code/mikltea/avatar.png`; no engine or agent Python changes are required.

## Generation prompt

Use case: stylized-concept. Asset type: an actual tiny Bomberman game avatar PNG for a character named mikltea. Create ONE charming bubble milk tea cup character as a very low resolution 30 x 30 pixel-art sprite. The game uses 30x30 RGBA sprites, so design on exactly a 30-by-30 logical pixel grid. Full character fits within the canvas with a 1-pixel transparent margin: recognizable tapered cup with milk tea, a sealed lid, a short thick drinking straw, several dark tapioca pearls visibly at the bottom, two tiny expressive dark eyes, and small feet so it reads as a game character. Classic old-school shaded pixel game art, crisp stepped silhouette and sparse highlights, a dark readable outline, a small restrained palette of natural milk-tea tan/cream and dark brown pearls. No text or logo. No scene, no drop shadow, no ground, no frame, no accessories beyond the cup and straw. Genuinely transparent background, NOT a drawn checkerboard. Intended to match the compact 30px sprite scale of the project's Crazy Turtle character: cute small recognizable character, gently shaded pixels, full body seen slightly from the front. Output exactly 30x30 pixels if possible; otherwise render the same coarse 30x30 design as an integer nearest-neighbor enlargement, without subpixel details, smoothing or antialiasing, centered square. This is a production sprite, not a mockup or sprite sheet.

## Export

The generator returned a larger RGBA image. FFmpeg cropped the surrounding transparent margin, scaled the character to 19 × 28 with nearest-neighbor sampling, and padded to the required 30 × 30 transparent canvas. Generated alpha is retained.

```bash
ffmpeg -i bubble_tea_source.png -vf 'crop=494:736:382:248,scale=19:28:flags=neighbor,pad=30:30:5:1:color=0x00000000,format=rgba' -frames:v 1 ../avatar.png
```

## Touch-up revision

The active `../avatar.png` is now the 30×30 RGBA touch-up, copied from `avatar_refined.png`. `avatar_preview.png` is its 20× nearest-neighbor preview. The pre-touch-up sprite is preserved in `avatar_before_touchup.png`.

The built-in image-generation tool restored eye glints and separated the tapioca pearls. Follow-up prompts requested background-only replacement with flat magenta for transparent export, then smaller pearls and wider gaps between their two rows. The selected generated source is `bubble_tea_touchup_final_source.png`. The export removes the magenta background/fringe, samples a 900×1110 window at (180,84) to 30×30 with nearest-neighbor scaling, and clears RGB in fully transparent pixels. This slightly widens the cup to improve detail readability. Further manual touch-up is left to the user as requested.
