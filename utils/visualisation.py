import pdfplumber

COLORS = {
    "Abstract":(255, 182, 193), "Author":(0, 0, 139), "Caption":(57, 230, 10),
    "Equation":(255, 0, 0),"Figure":(230, 51, 249),"Footer":(255, 255, 255),
    "List":(46, 33, 109),"Paragraph":(181, 196, 220),"Reference":(81, 142, 32),
    "Section":(24, 14, 248),"Table":(129, 252, 254),"Title":(97, 189, 251)
}

def draw_token(rect, im_origin):
    label = rect['label']
    color = COLORS.get(rect['label'], (0,0,0))
    bbox = (rect['x0'], rect['y0'], rect['x1'], rect['y1'])
    im_origin.draw_rect(bbox, 
                        fill=(color[0], color[1], color[2], 200), 
                        stroke=color, 
                        stroke_width=1)
    
def showpage(pdf_path, page=0, tokens=None):
    with pdfplumber.open(pdf_path) as doc:
        p0 = doc.pages[page - 1]
        im_origin = p0.to_image(resolution=300)

    if tokens is not None:
        for i, rect in tokens[tokens['page'] == page + 1].iterrows():
            draw_token(rect, im_origin=im_origin)

    im_origin.show()