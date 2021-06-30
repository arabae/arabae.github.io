# **Ara's DevBlog** 🤓

여러가지 공부하고, 정리하는 블로그입니다.

## Preview

[ARa's DevBlog](https://arabae.github.io/arabae.github.io/)

</br>

## How to start

- _config.yml example

```yaml
name: A Blog
author: Lightfish Zhang
url: https://lightfish-zhang.github.io
resume_site: https://lightfish-zhang.github.io
baseurl:
description: you website desc
github_username: lightfish-zhang
github: https://github.com/lightfish-zhang
plugins: [jekyll-paginate]
permalink: /:year-:month-:day-:title
paginate: 12
paginate_path: "/page/:num/"
exclude: ['README.md', 'Gemfile.lock', 'Gemfile', 'Rakefile']
highlighter: rouge
markdown: kramdown
comments :
  gitalk :
    clientID : xxx
    clientSecret : xxx
    repo : lightfish-zhang.github.io
    owner : lightfish-zhang
    admin : lightfish-zhang

```

- add your post in path `./_post`, format :

```md
---
layout: post
title: A Example Post
date:   1970-01-01 00:00:00 +0800
category: tutorial
thumbnail: /style/image/thumbnail.jpg
icon: book
---


* content
{:toc}

## sub title

page...

## about thumbnail

add the thumbnail url

## about icon

such as book, code, web, chat, note, game, link, design, image
```

some config about gitalk, please reference to [gitalk](https://github.com/gitalk/gitalk)

run `bundle install` and `jekyll server` to preview site on you computer, more question about jekyll, reference to [jekyll](http://jekyllrb.com)

## Thanks

- [jekyll](http://jekyllrb.com) git page engine
- [pinghsu](https://github.com/chakhsu/pinghsu), a typecho theme, it's a great design.
- [gitalk](https://github.com/gitalk/gitalk) git page comment engine, it depends on github issue.
- [smoothscroll](https://www.smoothscroll.net/mac/) SmoothScroll will give your mouse wheel (Finder, Safari, Chrome, etc.) buttery smooth scrolling

- [chakhsu](https://github.com/chakhsu)
- [lightfish-zhang](https://github.com/lightfish-zhang)
