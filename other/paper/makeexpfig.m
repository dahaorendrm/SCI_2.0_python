FOLDER = 'G:\My Drive\PHD_Research\SCI_2.0_python\video\mat\';
SAVEPATH = 'G:\My Drive\PHD_Research\SCI_2.0_python\video\fig\';
dataname = '4D_color_checker_motion';
meaidx = 30;

mkdir(SAVEPATH,[dataname]);

load([FOLDER,dataname,sprintf('%04d', meaidx-1),'.mat']);

imwrite(mea/max(mea(:)),[SAVEPATH,dataname,...
    '\measurement','.png'],'BitDepth',16);
spec = spec/max(spec(:));

%     % led
%     led = led/max(led(:));
%     for idx = 1:size(led,3)
%         imwrite(led(:,:,idx),[SAVEPATH,dataname,...
%             '\led',sprintf('%02d', idx),'.png'],'BitDepth',16);
%     end
%     % led complete
%     led_complete = led_complete/max(led_complete(:));
%     for idx1 = 1:size(led_complete,3)
%     for idx2 = 1:size(led_complete,4)
%         imwrite(led_complete(:,:,idx1,idx2),[SAVEPATH,dataname,...
%             '\led_complete',sprintf('%02d_%02d', idx1, idx2),'.png'],'BitDepth',16);
%     end
%     end
%     % rgb
%     rgb = rgb/max(rgb(:));
%     for idx = 1:size(rgb,4)
%         imwrite(rgb(:,:,:,idx),[SAVEPATH,dataname,...
%             '\rgb',sprintf('%02d', idx),'.png'],'BitDepth',16);
%     end
    % spec
    splist = [440:10:680];
    huelist = fliplr([0:0.0236:0.71]); % correspond to range 400-700nm
    
    for idx1 = 1:3:size(spec,3)
    for idx2 = 1:size(spec,4)
        temp = repmat(spec(:,:,idx1,idx2),[1 1 3]);
        temp = hueadjust(temp,huelist(idx1+4));
        imwrite(temp,[SAVEPATH,dataname,...
            '\spec',sprintf('%02d_%02d', idx1, idx2),'.png'],'BitDepth',16);
    end
    end


function img = hueadjust(img,hue)
    hsv = rgb2hsv(img);
    hsv(:,:,1) = repmat(hue,[256 256]);
    hsv(:,:,2) = repmat(1,[256 256]);
    img = hsv2rgb(hsv);
end


