clc;
close all;
clear;

% allItems = dir;
% patient_folders = allItems([allItems.isdir]);

target_subfolder = 'bright_blood/';
patient_folder = 'PAT29/';

if ~strcmp(patient_folder, '.') && ~strcmp(patient_folder, '..')
    brigth_blood_folder = fullfile(patient_folder, target_subfolder);
    fprintf('Processing: %s\n', brigth_blood_folder)

    files_names_dcm = {};
    cont_dcm = 1;

    filePattern_dcm = sprintf('%s\\*.dcm', brigth_blood_folder);
    baseFileNames_dcm = dir(filePattern_dcm);
    numberOfFiles_dcm = length(baseFileNames_dcm);

    if numberOfFiles_dcm >= 1
        for n = 1:numberOfFiles_dcm
            files_names_dcm{cont_dcm} = [brigth_blood_folder, filesep, baseFileNames_dcm(n).name];
            cont_dcm = cont_dcm + 1;
        end
    end

    info = dicominfo(files_names_dcm{1});
    imagen = zeros(info.Rows,info.Columns, size(files_names_dcm,2));
    voxel_size = [info.PixelSpacing',info.SliceThickness];
    fprintf('Rows: %d, Columns: %d\n', info.Rows, info.Columns);

    for n = 1:size(files_names_dcm,2)
        info = dicominfo(files_names_dcm{n});
        imagen(:,:,info.InstanceNumber) = double(dicomread(files_names_dcm{n}));
    end

    disp(voxel_size)
   
    slices = 16;
    img_list = cell(1, slices);

    % initial_point = [168.28, 183.15, 47.42];
    initial_point = [183.15, 168.28, 47.42];
    % final_point = [195.52, 290.33, 38.05];
    final_point = [290.33, 195.52, 38.05];
    normal = [0.245, 0.965, -0.084];

    distance = sqrt(sum((final_point - initial_point).^2));
    d = distance / (slices - 1);

    point = initial_point;
    adjusted_point = point; % ./ voxel_size;

    for n = 1:slices
        
        [B,x,y,z] = obliqueslice(imagen, adjusted_point, normal);

        img_list{n} = B;

        point = point + (d .* normal);
        adjusted_point = point; % ./ voxel_size;
    end
    
    figure
    imshow(img_list{1}, [])
    
    max_height = max(cellfun(@(img) size(img, 1), img_list));
    max_width = max(cellfun(@(img) size(img, 2), img_list));

    fprintf('Max height: %d, Max width: %d\n', max_height, max_width);

    % add padding
    for i = 1:slices
        img = img_list{i};
        [h, w] = size(img);
        padded_img = padarray(img, [max_height - h, max_width - w], 0, 'post');
        volume(1:max_height, 1:max_width, i) = padded_img;
    end

    save(fullfile(patient_folder,'volume.mat'), 'volume');

    figure
    montage((volume/max(volume(:))))

  
    % [B,x,y,z] = obliqueslice(imagen, adjusted_point, normal);
    % 
    % disp(size(B))
    % figure
    % imshow(B,[])
    % title('Slice in Image Plane')
    % 
    % point_1 = point + d .* normal;
    % disp(point_1);

    % figure
    % surf(x,y,z,B,'EdgeColor','None','HandleVisibility','off');
    % grid on
    % view([-38 12])
    % colormap(gray)
    % xlabel('x-axis')
    % ylabel('y-axis');
    % zlabel('z-axis');
    % title('Slice in 3-D Coordinate Space')
    % 
    % hold on
    % plot3(point(1),point(2),point(3),'or','MarkerFaceColor','r');
    % plot3(point(1)+[0 normal(1)],point(2)+[0 normal(2)],point(3)+[0 normal(3)], ...
    %     '-b','MarkerFaceColor','b');
    % hold off
    % legend('Point in the volume','Normal vector')
end
