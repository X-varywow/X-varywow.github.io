/**
 * 代码块增强插件（Docsify）
 * 1. 限制代码块显示高度，超出部分在块内滚动
 * 2. 右上角提供「复制」按钮
 *
 * 可选配置（写在 window.$docsify 里）：
 *   codeBlock: {
 *     maxHeight: '480px',   // 单个代码块最大高度
 *     onlyLang: ''          // 只处理某种语言，如 'txt'；留空表示全部
 *   }
 */
(function () {
    'use strict';

    var config = (window.$docsify && window.$docsify.codeBlock) || {};
    var MAX_HEIGHT = config.maxHeight || '';
    var ONLY_LANG = (config.onlyLang || '').toLowerCase();

    var TEXT = {
        copy: '复制',
        copied: '已复制',
        failed: '复制失败'
    };

    // 兼容非安全上下文（http / file）的复制方案
    function fallbackCopy(text) {
        var textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.setAttribute('readonly', '');
        textarea.style.cssText = 'position:fixed;top:0;left:0;width:1px;height:1px;opacity:0;';
        document.body.appendChild(textarea);
        textarea.select();
        textarea.setSelectionRange(0, textarea.value.length);

        var success = false;

        try {
            success = document.execCommand('copy');
        } catch (error) {
            success = false;
        }

        document.body.removeChild(textarea);

        return success;
    }

    function copyText(text) {
        if (navigator.clipboard && window.isSecureContext) {
            return navigator.clipboard.writeText(text).catch(function () {
                return fallbackCopy(text)
                    ? Promise.resolve()
                    : Promise.reject(new Error('Copy failed'));
            });
        }

        return fallbackCopy(text)
            ? Promise.resolve()
            : Promise.reject(new Error('Copy failed'));
    }

    function setButtonText(button, text) {
        button.textContent = text;

        window.clearTimeout(button._resetTimer);
        button._resetTimer = window.setTimeout(function () {
            button.textContent = TEXT.copy;
        }, 1500);
    }

    function enhance(pre) {
        if (pre.getAttribute('data-code-block') === '1') {
            return;
        }

        if (pre.parentNode && pre.parentNode.classList.contains('docsify-code-block')) {
            return;
        }

        var code = pre.querySelector('code');

        if (!code) {
            return;
        }

        var lang = pre.getAttribute('data-lang') || '';

        if (ONLY_LANG && lang.toLowerCase() !== ONLY_LANG) {
            return;
        }

        var wrapper = document.createElement('div');
        wrapper.className = 'docsify-code-block';

        if (MAX_HEIGHT) {
            wrapper.style.setProperty('--code-block-max-height', MAX_HEIGHT);
        }

        var bar = document.createElement('div');
        bar.className = 'docsify-code-block__bar';

        var langLabel = document.createElement('span');
        langLabel.className = 'docsify-code-block__lang';
        langLabel.textContent = lang;

        var button = document.createElement('button');
        button.type = 'button';
        button.className = 'docsify-code-block__copy';
        button.textContent = TEXT.copy;
        button.setAttribute('aria-label', '复制代码');

        button.addEventListener('click', function () {
            copyText(code.textContent || '').then(function () {
                setButtonText(button, TEXT.copied);
            }).catch(function () {
                setButtonText(button, TEXT.failed);
            });
        });

        bar.appendChild(langLabel);
        bar.appendChild(button);

        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(bar);
        wrapper.appendChild(pre);

        pre.setAttribute('data-code-block', '1');
    }

    function codeBlockPlugin(hook) {
        hook.doneEach(function () {
            var section = document.querySelector('.markdown-section');

            if (!section) {
                return;
            }

            var blocks = section.querySelectorAll('pre');

            for (var i = 0; i < blocks.length; i++) {
                enhance(blocks[i]);
            }
        });
    }

    window.$docsify = window.$docsify || {};
    window.$docsify.plugins = (window.$docsify.plugins || []).concat(codeBlockPlugin);
})();
