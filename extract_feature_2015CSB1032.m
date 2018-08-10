function [ extracted_feature ] = extract_feature_2015CSB1032(conv_net, image)
    img = single(image);
    meta_norm = imresize(img, conv_net.meta.normalization.imageSize(1:2));
    normalized_img = bsxfun(@minus, meta_norm, conv_net.meta.normalization.averageImage);
    conv_net.eval({'data', normalized_img});
    extracted_feature = reshape(conv_net.vars(152).value, 1, 1024);
end