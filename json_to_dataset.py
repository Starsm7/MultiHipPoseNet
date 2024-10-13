import base64
import json
import os
import os.path as osp
import numpy as np
import PIL.Image
from labelme import utils


# Convert a json file annotated with labelme into a masked png.


if __name__ == '__main__':
    jpgs_path   = "data/JPEGImages"
    pngs_path   = "data/SegmentationClass"
    classes     = ["_background_", "股骨头", "大转子", "盂唇", "软骨性髋臼顶", "滑膜皱襞", "近端软骨膜","Y型软骨"]

    count = os.listdir("./data/before")
    for i in range(0, len(count)):
        path = os.path.join("./data/before", count[i])
        print(path)

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path, encoding='utf-8'))
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            Data=[]

            for shape in data['shapes']:
                if shape["shape_type"] == "polygon":
                    if shape['label'] in classes:
                        label_name = shape['label']
                        if label_name in label_name_to_value:
                            label_value = label_name_to_value[label_name]
                        else:
                            label_value = len(label_name_to_value)
                            label_name_to_value[label_name] = label_value
                    else:
                        Data.append(shape)
                else:
                    Data.append(shape)

            data['shapes'] = [item for item in data['shapes'] if item not in Data]

            assert len(data['shapes'])==7
            
            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))
            
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            
                
            PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0]+'.jpg'))
            print(osp.join(jpgs_path, count[i].split(".")[0]+'.jpg'))

            new = np.zeros([np.shape(img)[0], np.shape(img)[1]], dtype=np.uint8)
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                new = new + index_all * (np.array(lbl) == index_json)[0]

            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0]+'.png'), new)
            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')
