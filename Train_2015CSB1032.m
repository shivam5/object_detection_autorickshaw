% Loading the JSON file
disp('Loading labels file');
json_file_path = 'auto_det_chal_train_7oct/bbs/bbs.json';
json_file= fopen(json_file_path);
json_str = char(fread(json_file,inf)');
fclose(json_file);
label_data = JSON.parse(json_str);


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


if (exist('SVM_models.mat', 'file') == 0 )
    if (exist('train_features.mat', 'file') == 0 )
        disp('Creating train features');
        
        n_images = 600;

        height = 140;
        width = 140;

        features = zeros(1024);

        image_path = 'auto_det_chal_train_7oct/images/';
        for i = 1:n_images
            disp(i);
            img_filename = strcat(image_path,strcat(int2str(i-1),'.jpg'));
            img=imread(img_filename);
            for j = 1:size(label_data{i}, 2)
                if(i==286 && j==1)
                    continue;
                end
                raw_bbox=cell2mat([label_data{i}{j}{:}]);
                x_coords = [raw_bbox(1),raw_bbox(3),raw_bbox(5),raw_bbox(7)];
                x1 = floor(min(x_coords));
                x2 = ceil(max(x_coords));
                y_coords = [raw_bbox(2),raw_bbox(4),raw_bbox(6),raw_bbox(8)];
                y1 = floor(min(y_coords));
                y2 = ceil(max(y_coords));
                bbox_dim = [x1, y1, x2-x1, y2-y1];
                cropped_bbox = imcrop(img, bbox_dim);
                if (bbox_dim(4)+1 == size(cropped_bbox,1) && bbox_dim(3)+1 == size(cropped_bbox,2))
                    for h=1:50:size(cropped_bbox, 1)-height+1
                        for w=1:50:size(cropped_bbox,2)-width+1
                            box = [w h width-1 height-1];
                            cropped = imcrop(cropped_bbox, box);
                            features = [features; extract_feature_2015CSB1032(convnet, cropped) ];

                        end
                    end
                end
                
            end
        end
        features = features(2:size(features,1), :);

        img_dims = [height, width];

        save('train_features.mat', 'features', 'img_dims');
    end

    disp('Loading train features files');
    S = load('train_features.mat');

    disp('Training SVMS');
    Y = ones(size(S.features, 1), 1);
    
    SVMModel = fitcsvm(S.features, Y, 'KernelScale','auto','Standardize',true, 'OutlierFraction',0.03,  'ClassNames', [1]);
%     SVMModel = svmtrain(double(Y), double(S.features), '-s 2 -t 2');
    img_dims = [140, 140];
    save('SVM_models.mat', 'SVMModel', 'img_dims');
end