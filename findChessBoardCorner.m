function [] = findChessBoardCorner( mydir )
format short;
allJpgFiles=dir([mydir,'*.jpg']);
for i = 1: length(allJpgFiles)
    imageFileNames{i} = allJpgFiles(i).name;
    imageFilePaths{i} = fullfile(mydir, allJpgFiles(i).name);
    disp(imageFileNames{i});
end
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFilePaths);
%从左到右从上到下，顺序为左上、左下、右上、右下
count = 0;
fid = fopen([mydir 'calibration\\UsedImages.dat'], 'w');
for i = 1: length(imageFileNames)
    if imagesUsed(i)
        count = count + 1;
        usedFileNames{i} = imageFileNames{i};
        fprintf(fid, '%s ', imageFileNames{i});
    end
end
fclose(fid);

fid = fopen([mydir 'calibration\\BoardSize.dat'], 'w');
fprintf(fid, '%d %d', boardSize(1) - 1, boardSize(2) - 1);
%先高后宽
fclose(fid);

for i = 1: count
    fid = fopen([mydir 'calibration\\' usedFileNames{i} '.dat'], 'w');
    for j = 1: (boardSize(1) - 1) * (boardSize(2) - 1)
        fprintf(fid, '%5f %5f\n', imagePoints(j, 1, i), imagePoints(j, 2, i));
    end
    %先高后宽
    fclose(fid);
end
end

