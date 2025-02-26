java c
EECS 442 Practice Exam, Winter 2025 
Problem 1 Machine   Learning   Fundamentals (16 points) 
(a) (4 points) List   two   methods   to   prevent   underfitting   and   overfitting   respectively.
(b) (4 points) How   does   increasing   or   decreasing   a   linear   model’s   regularization   parameter (the   λ   term   in   L2   regularization)   affect   the   learned   model’s   complexity?
(c) (4 points) Explain   the   roles   of   training,   validation,   and   testing   datasets   in   the   machine learning   workflow.   Why   is   it   important   to   separate   these   datasets?
(d) (4 points) Explain   the   purpose   of cross-entropy   loss   in   classification   tasks.    Why   is   it   preferred   over   MSE   (mean-squared-error)   loss   in   classification?
Problem 2 Image   Filtering (16 points) 
Given   the   original   5   × 5   grayscale   image   matrix:

A   noise   spike   is   present   at   the   center   of   the   matrix   (200).   Your   goal   is   to   reduce   this   noise while   preserving   other   details.(a) (4 points) Median Filtering (3   × 3   filter   window   with   zero   padding)   Apply   a   3   × 3   median   filter   to   the   entire   image   using   zero-padding   to   keep   the   output   matrix   the   same   size (5   × 5)   as   the   input.   Please   provide   the   filtered   output   matrix.(b) (4 points) Gaussian Filtering (3 × 3   filter   window   without   padding)   Apply   a   Gaussian filter   with   a   3   × 3   kernel   using   the   following   values   (not   normalized),   without   any   padding,   so   that   the   output   matrix   size   is   reduced   to   3   × 3:

After   filtering,   normalize   by   dividing   each   pixel   value   by   16.   Please   provide   the   resulting   3   × 3   output   matrix.(c) (2 points) Is the above Gaussian filter kernel separable? If so,   pleased   write   the   two   1D   rectangular   filter   kernels   that   can   be   used   to   reconstruct   the   3   × 3   Gaussian   filter.   If not,   please   give   the   reason.(d) (4 points) Bilateral Filtering (3   × 3   filter window without   padding)   Apply   a   bilateral   filter   with   a   3   × 3   window   centered   on   each   pixel.    Use   the   Gaussian   kernel   from   Question (b)   for   spatial   weighting.   For   intensity   weighting,   we   will   use   a   simplified   scheme   that   makes computation   easier:   we   will   use   an   intensity   threshold   of 30,   which   means   that   pixels   with   at   most   an   intensity   difference   of   30   from   the   center   pixel   would   contribute.   For   example,   if   the center   pixel   is   50,   only   neighboring   pixels   within   the   intensity   range   of   [20   ,   80]   will   contribute to   the   filtered   results.    More   precisely,   the   equation   for   this   simplified   bilateral   filter   is   as   follows:
where:
• Ifiltered(x)   is the   intensity   of the   filtered   image   at   pixel   x;
• N(x)   is the   spatial   neighborhood   of pixel   x;
• I(x′)   is the   intensity   of   the   neighboring   pixel   x′ ;
•    Gσs         is   the   spatial   Gaussian   kernel   with   spatial   standard   deviation σs.   For   simplification,   we   just   use   the   Gaussian   kernel   from   Question   (b)   as   an   approximation   for   Gσs;
• W(x)   is   the   normalization   factor:
• δ(I(x), I(x′ ))   is   the   indicator   function   enforcing   the   intensity   difference threshold:
This   equation   combines   both   spatial   and   intensity  代 写EECS 442 Practice Exam, Winter 2025SPSS
代做程序编程语言 constraints   for   bilateral   filtering,   using   the   given   Gaussian kernel   and intensity threshold.    Please provide the resulting   3   ×   3   output   matrix   after   applying   this   bilateral   filtering   using   a   3   ×   3   filter   window   without   padding.
(e) (2 points) Compare the   output   matrices   from the three   filters.    Discuss   in   a   few   sentences:   which   filter   best   reduced   the   noise   spike   while   preserving   details   and   explain   why.
Problem 3 Fourier    Transform. (16 points) 
(a) (2 points) If   we   rotate   an   image   clockwise   by   90。,   what   effect   does   this   have   on   its Fourier   transform?
a)   The   Fourier   transform   rotates   90。 clockwise.
b)   The   Fourier   transform   rotates   90。 counter-clockwise.
c)   The   Fourier   transform   rotates   180。.
d)   No   general   statement   can   be   made   about   the   Fourier   transform.
(b) (2 points) If   we   scale   an   image   by   a   factor   of   2,   doubling   its   size,   what   effect   does   this have   on   its   Fourier   transform?
a)   The   Fourier   transform   scales   by   a   factor   of   2.
b) The Fourier transform. scales by a factor of 2/1.
c)   The   Fourier   transform   is   unaffected.
d)   No   general   statement   can   be   made   about   the   Fourier   transform.
(c) (12 points) Please   match   the   following   images   to   their   respective   Fourier   transforms.
a)                       b)                           c)                           d)                           e)                           f)                        

Problem 4 Backpropagation (16 points) Recall   that   a   neural   network   can   be   represented   as   a   computation   graph,   enabling   us   to   systematically   compute   its   gradients.   For   example,   Figure 1   is   an   example   of the   equation   f(x,y) = x + y.    The   corresponding   code   for   the   forward   and   backward   of   this   diagram   is   also shown   below.

Figure   1:   Computation   graph   for   f(x,y) = x + y
1         def         f(x,      y):
2                                           ###forward         pass   ###
3                                                      L      =       x       +       y
4
5                                                      ###backward         pass   ###
6                                                 grad   _L         =         1
7                                              grad_x         =         1         *         grad_L
8                                              grad_y         =         1         *         grad_L
9                                                      return    L,         (grad_x   ,         grad_y)
Figure 2   is   a   computation   graph   for   function   f(a,b,c,d).


Figure   2:   Computation   graph
(a) (2 points) Please   write   down   the   mathematical   formula   for   f(a,b,c,d).
(b) (4 points) Please   implement   the   code   for   forward   and   backward   pass   of computation   graph   in   (a).
(c) (7 points) Please   draw   the   computation   graph   and   implement   the   code   for   forward   and backward   pass   of function  
Note:    Please      use   the   following   operations:    +, ×   , −   ,   +1, ×(−1),exp, x/1   .
(d) (3 points) Why   might   Stochastic   Gradient   Descent   (SGD)   be   more   effective   than   Batch Gradient   Descent   (BGD)   when   training   neural   networks? 

         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
