FOLDER = 'G:\My Drive\PHD_Research\SCI_2.0_python\video\mat\';
SAVEPATH = 'G:\My Drive\PHD_Research\SCI_2.0_python\video\seq\';
dataname = '116fps_toys';
range = [0,119];
    
mkdir(SAVEPATH,dataname)
for sidx=range(1)+1:(range(2)+1)
    load([FOLDER,dataname,sprintf('%04d', sidx-1),'.mat']);
    for fidx = 1:24
        %rgb = flip(rgb,3);
        frame = makeaframe(sidx,fidx, mea,led,led_complete,spec,rgb);
        imwrite(frame,[SAVEPATH,dataname,'\',sprintf('%04d', (sidx-1)*24+fidx),'.png'],'BitDepth',16);
        % save frame        
    end
end

    
%% Test a frame   
function img = hueadjust(img,hue)
    hsv = rgb2hsv(img);
    hsv(:,:,1) = repmat(hue,[128 128]);
    hsv(:,:,2) = repmat(1,[128 128]);
    img = hsv2rgb(hsv);
end

function new_frame = makeaframe(sidx,fidx, mea,led,led_complete,spec,rgb)
size_im = 256;
size_gap = 10;
%fidx = 1;
new_frame = ones(720,1280,3); % 720p
% Add the measurement
new_frame = insertText(new_frame,[0 0],'Measurements','FontSize',24,'BoxColor','white','TextColor','black'); 
por = 40;
poc = size_gap;
new_frame(por:por+size_im-1,poc:poc+size_im-1,:) = repmat(mea/max(mea(:)),[1 1 3]);

% Add the result of S1
por = 40;
poc = size_gap*2+size_im;
new_frame = insertText(new_frame,[poc 0],'S1 Result','FontSize',24,'BoxColor','white','TextColor','black'); 
temp = led(:,:,fidx);
new_frame(por:por+size_im-1,poc:poc+size_im-1,:) = repmat(temp/max(led(:)),[1 1 3]);
new_frame = insertText(new_frame,[poc por],['LED ',num2str(mod(fidx-1,8)+1)],'FontSize',20,'BoxColor','white','BoxOpacity',0,'TextColor',[0,1,0]); 

% Add the result of S2
por = 40;
poc = size_gap*3+size_im*2;
s2r = zeros(128,128,8);
for lidx = 1:8
    s2r(:,:,lidx) = imresize(led_complete(:,:,lidx,fidx),0.5);
end
new_frame = insertText(new_frame,[poc 0],'S2 Result','FontSize',24,'BoxColor','white','TextColor','black'); 
for lidx = 1:8
    if lidx ==1
    elseif lidx == 5
        por = por + size_im/2;
        poc = size_gap*3+size_im*2;
    else
        poc = poc+size_im/2;
    end
    new_frame(por:por+size_im/2-1,poc:poc+size_im/2-1,:) = repmat(s2r(:,:,lidx)/max(s2r(:)),[1 1 3]);
    new_frame = insertText(new_frame,[poc por],['LED ',num2str(lidx)],'FontSize',10,'BoxColor','white','BoxOpacity',0,'TextColor',[0,1,0]); 
end

% Add the result of S3
splist = [440:10:680];
huelist = fliplr([0:0.0236:0.71]); % correspond to range 400-700nm
por = 40*2+size_im;
poc = 1;
s3r = zeros(128,128,25);
for lidx = 1:25
    s3r(:,:,lidx) = imresize(spec(:,:,lidx,fidx),0.5);
end
new_frame = insertText(new_frame,[poc size_im+40],'S3 Result','FontSize',24,'BoxColor','white','TextColor','black'); 
for lidx = 1:25
    if lidx ==1
    elseif lidx == 9 ||  lidx == 17
        por = por + size_im/2;
        poc = 1;
    else
        poc = poc+size_im/2;
    end
    temp = repmat(s3r(:,:,lidx)/max(spec(:)),[1 1 3]);
    temp = hueadjust(temp,huelist(lidx+4));
    new_frame(por:por+size_im/2-1,poc:poc+size_im/2-1,:) = temp;
    new_frame = insertText(new_frame,[poc por],[num2str(splist(lidx)),' nm'],'FontSize',10,'BoxColor','white','BoxOpacity',0,'TextColor',[0,1,0]); 
end

% Add the result of RGB
por = 40*2+size_im;
poc = 1024;
% new_frame = insertText(new_frame,[size_im+40 poc],'Recovered RGB','FontSize',24,'BoxColor','white','TextColor','black');  
new_frame(por:por+size_im-1,poc:poc+size_im-1,:) = rgb(:,:,:,fidx);
new_frame = insertText(new_frame,[poc por],'Recovered RGB','FontSize',20,'BoxColor','white','BoxOpacity',0,'TextColor',[0,1,0]); 

% Add frame info
new_frame = insertText(new_frame,[1064 85],['Snapshot: ',num2str(sidx)],'FontSize',24,'BoxColor','yellow','BoxOpacity',0.4,'TextColor','black');  
new_frame = insertText(new_frame,[1064 130],['Time: ',num2str((sidx-1)*24+fidx)],'FontSize',24,'BoxColor','yellow','BoxOpacity',0.4,'TextColor','black');  
%imshow(new_frame);
end





