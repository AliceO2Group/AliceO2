%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% customization for the ALICE Technical Design and Physics Performance Reports 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\fileversion{8}
\def\filedate{1998/07/31}
\def\docdate{1994/01/10}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{alicetdr}[\filedate]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\PassOptionsToClass{a4paper,twoside,11pt}{report}
\ProcessOptions
\LoadClass{report}
\RequirePackage{graphicx}
\RequirePackage{pifont}
\RequirePackage{array,tabularx}
\RequirePackage{verbatim}
\RequirePackage{graphicx,epsfig}
\RequirePackage{varioref}
\RequirePackage{subfigure}
\RequirePackage{amsmath,amssymb}
\RequirePackage{cite}
%\RequirePackage{bibunits}
\RequirePackage{here}
\RequirePackage{xspace}
\RequirePackage{calc,ifthen}
\RequirePackage{makeidx}
\RequirePackage{rotating}
%\RequirePackage{mathptm}
\RequirePackage{multirow}
\RequirePackage{pstricks,pst-plot}
\renewcommand{\rmdefault}{ptm}

%%%  Comment next lines to get black on white %%%
%\let \oldref=\ref
%\renewcommand{\ref}[1]{{\red\oldref{#1}}}

\pagestyle{headings}
% typeset caption contents in small
\long\def\@makecaption#1#2{%
   \vskip 10\p@
   \setbox\@tempboxa\hbox{{\sl\bf #1:} {\small\rm #2}}%
   \ifdim \wd\@tempboxa >\hsize
       {\sl\bf #1:} {\small\rm #2}\par
     \else
       \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
   \fi}

\setcounter{dbltopnumber}{3}
\renewcommand{\dbltopfraction}{.9}
\renewcommand{\dblfloatpagefraction}{.85}

\def\@makechapterhead#1{%
 %  \twocolumn[%
 \noindent\begin{minipage}{\textwidth}%
                 \raggedright
 \huge\bf  \thechapter\quad #1\par
            \vspace*{-3mm}%
            \rule{\textwidth}{0.8pt}%
            \vspace*{+8mm}%
       \end{minipage}%
  % ]%
}
 
\def\@makeschapterhead#1{%
 %  \twocolumn[%
\noindent\begin{minipage}{\textwidth}%
                 \raggedright
                \huge\bf #1\par
            \vspace*{-3mm}%
            \rule{\textwidth}{0.8pt}%
            \vspace*{+8mm}%
       \end{minipage}%
 %  ]%
}
 
\def\chapter{\thispagestyle{plain}\secdef\@chapter\@schapter}
 
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
        \refstepcounter{chapter}%
        \typeout{\@chapapp\space\thechapter.}%
        \addcontentsline{toc}{chapter}{\protect
        \numberline{\thechapter}#1}\else
      \addcontentsline{toc}{chapter}{#1}\fi
   \chaptermark{#1}%
   \addtocontents{lof}%
       {\protect\addvspace{10\p@}}
   \addtocontents{lot}%
       {\protect\addvspace{10\p@}}
   \@makechapterhead{#1}
}

\def\Chapter#1{\addcontentsline{toc}{chapter}{#1}%
       \markboth{}{}\noindent
       \begin{minipage}{\textwidth}%
                 \raggedright
                \huge\bf #1\par
            \vspace*{-3mm}%
            \rule{\textwidth}{0.8pt}%
            \vspace*{+8mm}%
       \end{minipage}%
       \thispagestyle{plain}%
}

\def\@schapter#1{\@makeschapterhead{#1}}
 
\def\section{\@startsection {section}{1}{\z@}{-3.5ex plus-1ex minus
    -.2ex}{2.3ex plus.2ex}{\reset@font\Large\raggedright\bf}}
\def\subsection{\@startsection{subsection}{2}{\z@}{-3.25ex plus-1ex
    minus-.2ex}{1.5ex plus.2ex}{\reset@font\large\raggedright\bf}}
\def\subsubsection{\@startsection{subsubsection}{3}{\z@}{-3.25ex plus
    -1ex minus-.2ex}{1.5ex plus.2ex}{\reset@font\normalsize
    \raggedright\bf}}
\def\paragraph{\@startsection
    {paragraph}{4}{\z@}{3.25ex plus1ex minus.2ex}{-1em}{\reset@font
    \normalsize \bf}}
\def\subparagraph{\@startsection
     {subparagraph}{4}{\parindent}{3.25ex plus1ex minus
     .2ex}{-1em}{\reset@font\normalsize\bf}}
 
\setcounter{secnumdepth}{3}
% allows 3 levels of sections, e.g. '5.3.2 The Blablabla'
% sequence: \section \subsection \subsubsection \paragraph

\def\ps@headings{\let\@mkboth\markboth
  \def\@oddfoot{}\def\@evenfoot{}%       No feet.
  \def\@evenhead{\upshape\thepage\hfil \itshape\leftmark}%        Left heading.
  \def\@oddhead{{\itshape\rightmark}\hfil \upshape\thepage}% Right heading.
  \def\chaptermark##1{\markboth {%\uppercase
{%
  \thechapter\hskip 1em\relax ##1}}{}}%
  \def\sectionmark##1{\markright {%
        \thesection\hskip 1em\relax ##1}}}
\def\ps@plain{\let\@mkboth\markboth
  \def\@oddfoot{}\def\@evenfoot{}%       No feet.
  \def\@evenhead{\upshape\thepage\hfill}%                     Left heading.
  \def\@oddhead{\hfill \upshape\thepage}% Right heading.
  \def\chaptermark##1{\markboth {%\uppercase
{%
   % \thechapter\hskip 1em\relax ##1}}{}}%
  \thechapter\relax ##1}}{}}%
  \def\sectionmark##1{\markright {%
   %       \thesection\hskip 1em\relax ##1}}}
  \thesection\relax ##1}}}
\ps@headings
\pagenumbering{arabic}

\setlength{\voffset}{-15mm}
\setlength{\oddsidemargin}{15pt}        %%%%
\setlength{\evensidemargin}{15pt}      %%%%
\setlength{\textwidth}{160mm}
\setlength{\textheight}{240mm}

\def\bibname{References}
\def\thebibliography#1{\par\list
  {\@biblabel{\arabic{enumiv}}}{\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \def\theenumiv{\arabic{enumiv}}}%
    \def\newblock{\hskip .11em plus.33em minus.07em}%
    \sloppy\clubpenalty4000\widowpenalty4000
    \sfcode`\.=1000\relax}
 

\renewcommand*\l@chapter[2]{%
   %mg\addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
}

% adjust indent for table of content
\renewcommand{\l@section}{\@dottedtocline{1}{0.6em}{2.9em}}
\renewcommand{\l@subsection}{\@dottedtocline{2}{1.2em}{3.8em}}
\renewcommand{\l@subsubsection}{\@dottedtocline{3}{2.5em}{4.0em}}
\renewcommand{\l@paragraph}{\@dottedtocline{4}{3.5em}{4.5em}}
\renewcommand{\l@subparagraph}{\@dottedtocline{5}{4.5em}{5.0em}}
 
% adjust spacing between rows in tables
\renewcommand{\arraystretch}{1.05}

%\def\tableofcontents{%
%\chapter*{Table of Contents}\pagestyle{empty}\@starttoc{toc}}

\endinput
 


