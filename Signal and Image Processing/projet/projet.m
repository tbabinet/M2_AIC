close all;
%% ouverture image
im = imread("barbara.jpg");
imshow(im);
figure();
im = rgb2gray(im);

%% convolution
mask = ones(5,5);
imconv = conv2(im, mask, 'same');

max_ = max(max(imconv));
imconv = 255*(imconv/max_);
imconv = uint8(imconv);
imshow(imconv);
figure();

%% translation
im_trans_1 = imt