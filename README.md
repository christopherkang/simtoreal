# simtoreal

## Christopher Kang under William Agnew

## How to use

Currently, work is on the `randomize.py` file. Go into `point_mass.xml`, and add as many textures as necessary in, making sure to label them with different names. They should be similar to `<texture name="sample" type="cube" file="{}"/>`, with the `{}` denoting that it is open to randomization. (This was implemented with Python string formatting; you could use `{1}`, `{2}` etc. to have a different order when the paths are provided).

Next, go into `randomize.py` and update the `_TEXTUREPATHS` with filepaths to the PNGs to use. The format should be a nested list of arrays, where each array provides a list of possible textures to be used at the texture. For example:

```python
[
    ["dog.png", "cat.png"],
    ["carpet.png", "marble.png"]
]
```

In the above example, the first texture will choose between the `dog` and `cat` PNGs, while the second texture will choose between the `carpet` and `marble` PNGs. (These are chosen equally randomly, though this can be customized).
