% % Loading the JSON file
% disp('Loading labels file');
% json_file_path = 'auto_det_chal_train_7oct/bbs/bbs.json';
% json_file= fopen(json_file_path);
% json_str = char(fread(json_file,inf)');
% fclose(json_file);
% label_data = JSON.parse(json_str);

S = load('SVM_models.mat');
height = floor(S.img_dims(1));
width = floor(S.img_dims(2));
SVMModel = S.SVMModel;

run matconvnet-1.0-beta25/matlab/vl_setupnn;
if (exist('imagenet-googlenet-dag.mat', 'file') == 0 )
    disp('Downloading pre-trained model');
    urlwrite(...
    'http://www.vlfeat.org/matconvnet/models/imagenet-googlenet-dag.mat', ...
    'imagenet-googlenet-dag.mat') ;
end
convnet = dagnn.DagNN.loadobj(load('imagenet-googlenet-dag.mat'));    
convnet.mode = 'test';
convnet.conserveMemory = 0;

image_path = 'auto_det_chal_train_7oct/images/';
save_path = 'outputs/';
bounding_boxes = {};
for i = 601:800
    img_filename = strcat(image_path,strcat(int2str(i-1),'.jpg'));
    save_file = strcat(save_path,strcat(int2str(i-1),'.jpg'));
    img = imread(img_filename);
    original_size = size(img);
    img = imresize(img, [height*2 NaN]);
%     figure;
%     imshow(img);
%     hold on;
    new_size = size(img);
    transform = original_size./new_size;

    bboxes = [0,0,0,0];
    scores = [0];
    
    for h=1:10:size(img, 1)-height+1
        for w=1:10:size(img,2)-width+1
            box = [w h width-1 height-1];
            cropped = imcrop(img, box);
            feature = extract_feature_2015CSB1032(convnet, cropped);
            [label,score] = predict(SVMModel, feature);
            if (label==1 && score>500)
                bboxes = [bboxes; box];
                scores = [scores; score];
            end
        end
    end
    bboxes = bboxes(2:size(bboxes,1),:);
    scores = scores(2:size(scores,1),:);
    scores = normc(scores.^3);
    [selectedboxes, selectedscores] = selectStrongestBbox(bboxes, scores, 'OverlapThreshold', 0.2);
    
    final = imresize(img, [original_size(1) original_size(2)]);
    selectedboxes(:,1) = selectedboxes(:,1).*transform(1);
    selectedboxes(:,2) = selectedboxes(:,2).*transform(2);
    selectedboxes(:,3) = selectedboxes(:,3).*transform(1);
    selectedboxes(:,4) = selectedboxes(:,4).*transform(2);

    heat_map = zeros(original_size(1), original_size(2));
    for it=1:size(selectedboxes,1)
        heat_map(selectedboxes(it,2):selectedboxes(it,2)+selectedboxes(it,4), selectedboxes(it,1):selectedboxes(it,1)+selectedboxes(it,3))=heat_map(selectedboxes(it,2):selectedboxes(it,2)+selectedboxes(it,4), selectedboxes(it,1):selectedboxes(it,1)+selectedboxes(it,3))+0.1;
    end
    selectedboxes=[];    
    heat_map = heat_map>0;
    new_boundaries = bwboundaries(heat_map);
        
    for it = 1:size(new_boundaries,1)
        one_box = new_boundaries{it};
        y_coords = one_box(:,1);
        y1 = min(y_coords);
        y2 = max(y_coords);
        x_coords = one_box(:,2);
        x1 = min(x_coords);
        x2 = max(x_coords);
        bbox = [x1+20 y1+20 x2-x1-40 y2-y1-40];
        selectedboxes = [selectedboxes ; bbox];
    end
    
    labs = linspace(1, size(selectedboxes, 1), size(selectedboxes, 1))';
    with_boxes_image2 = insertObjectAnnotation(final,'rectangle',selectedboxes, cellstr(num2str(labs)), 'Color','r');
    imwrite(with_boxes_image2, save_file);
    bounding_boxes{i-600} = selectedboxes;
    
end
save('test_output.mat','bounding_boxes');