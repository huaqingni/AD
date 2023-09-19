a=ones(1,720);
b=[3
5
15
33
71
73
89
90
91
94
109
111
117
135
170
171
173
189
193
251
253
262
265
275
278
315
322
323
324
351
370
393
400
408
414
418
435
444
473
476
480
487
489
493
503
520
550
635
653
655
];
for i=1:50
    a(b(i))=0;
end
c=reshape(a,90,8);
[r, l] = size(c);                          % Get the matrix size
imagesc((1:l)+0.5, (1:r)+0.5, c);          % Plot the image
colormap(gray);                              % Use a gray colormap
axis equal                                   % Make axes grid sizes equal
set(gca, 'XTick', 1:(l+1), 'YTick', 1:(r+1), ...  % Change some axes properties
         'XLim', [1 l+1], 'YLim', [1 r+1], ...
         'GridLineStyle', '-', 'XGrid', 'on', 'YGrid', 'on');