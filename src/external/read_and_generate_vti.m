clc;
close all;
clear;

allItems = dir;
patient_folders = allItems([allItems.isdir]);

target_subfolder = 'bright_blood/';

for i = 1:length(patient_folders)
    patient_folder = patient_folders(i).name;
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
        for n = 1:size(files_names_dcm,2)
            info = dicominfo(files_names_dcm{n});
            imagen(:,:,info.InstanceNumber) = double(dicomread(files_names_dcm{n}));
        end

        mkdir([patient_folder, filesep, 'vti_brightblood']);

        name = 'volume_brightblood.vti';                
        fileID = fopen([patient_folder, filesep, 'vti_brightblood', filesep, name],'w');
        fprintf(fileID,'%s\n','<?xml version="1.0"?>');
        fprintf(fileID,'%s\n','<VTKFile type="ImageData" version="0.1">');
        fprintf(fileID,'%s\n',['<ImageData WholeExtent="1 ',num2str(size(imagen,1)),' 1 ',num2str(size(imagen,2)),' 1 ',num2str(size(imagen,3)),'" Origin="0 0 0" Spacing="',num2str(voxel_size(1)),' ',num2str(voxel_size(2)),' ',num2str(voxel_size(3)),'" >']);
        fprintf(fileID,'%s\n',['<Piece Extent="1 ',num2str(size(imagen,1)),' 1 ',num2str(size(imagen,2)),' 1 ',num2str(size(imagen,3)),'" >']);

        fprintf(fileID,'%s\n','<PointData Scalars="Imagen" Vectors="" Tensor="">');
        fprintf(fileID,'%s\n','<DataArray type="Float32" Name="Imagen" format="ascii">');
        fprintf(fileID,'%f\n',imagen(:)');
        fprintf(fileID,'%s\n','</DataArray>');
        fprintf(fileID,'%s\n','</PointData>');
        fprintf(fileID,'%s\n','<CellData>');
        fprintf(fileID,'%s\n','</CellData>');
        fprintf(fileID,'%s\n','</Piece>');
        fprintf(fileID,'%s\n','</ImageData>');
        fprintf(fileID,'%s\n','</VTKFile>');
        fclose(fileID);
    end
end