% 打开原始文件以进行读取
fileID = fopen('Node_AAL90.txt', 'r');

% 打开新文件以进行写入
newFileID = fopen('region.txt', 'w');

% 定义需要保存的行数
desiredLines = [16 12 76 74 10 15 25 31 42 8 47 30 68 29 27 63 72 38]; % 以此处为例，保存第2、4和6行

% 初始化计数器
lineCounter = 1;

% 逐行读取原始文件内容
while ~feof(fileID)
    % 读取当前行
    currentLine = fgetl(fileID);
    
    % 如果当前行是需要保存的行之一，则将其写入新文件
    if ismember(lineCounter, desiredLines)
        fprintf(newFileID, '%s\n', currentLine);
    end
    
    % 增加行计数器
    lineCounter = lineCounter + 1;
end

% 关闭文件
fclose(fileID);
fclose(newFileID);
