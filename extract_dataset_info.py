import docx

doc = docx.Document('/home/david/srtp/mcnn-pytorch-train/Documents/2022115902冯宇捷毕业设计论文（利兹）.docx')
for p in doc.paragraphs:
    text = p.text.strip()
    if 'dataset' in text.lower() or 'images' in text.lower():
        print(text)
