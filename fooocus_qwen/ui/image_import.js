() => {
    /* Перевод HEIC, RAW, TIFF, PSD, JPEG XL, EXR, SVG… в PNG до того, как
       файл увидит поле Gradio или кисть правки (см. image_import.py).
       Перехват — в фазе захвата на window: он срабатывает раньше обработчиков
       полей, поэтому сами поля о форматах ничего не знают. */
    if (window.__fqImport) return [];
    const NATIVE = __NATIVE__;
    const CONVERTED = __CONVERTED__;
    const PATH = __PATH__;
    const TEXT = __TEXT__;
    const NATIVE_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif', 'image/bmp'];
    // Самая длинная сторона растра SVG: вектор масштабируется без потерь, а
    // модели больше 2K не нужно.
    const SVG_SIDE = 2048;

    const langIndex = () => {
        const button = document.querySelector('.qs-lang');
        return button && button.innerText.trim().toLowerCase() === 'ru' ? 0 : 1;
    };
    const say = (key, values) => TEXT[key][langIndex()].replace(/\{(\w+)\}/g, (_, k) => values[k] ?? '');
    const extension = name => {
        const dot = (name || '').lastIndexOf('.');
        return dot < 0 ? '' : name.slice(dot).toLowerCase();
    };
    const stem = name => {
        const dot = (name || '').lastIndexOf('.');
        return (dot <= 0 ? name : name.slice(0, dot)) || 'image';
    };
    // Переводить ли: по расширению, а без знакомого расширения — по типу
    // (картинка, которую браузер не обещал показать).
    const needs = file => {
        if (!file) return false;
        const ext = extension(file.name);
        if (NATIVE.includes(ext)) return false;
        if (CONVERTED.includes(ext)) return true;
        return Boolean(file.type) && file.type.startsWith('image/') && !NATIVE_TYPES.includes(file.type);
    };
    const rootUrl = () => {
        const root = (window.gradio_config && window.gradio_config.root)
            || `${window.location.origin}${window.location.pathname}`;
        return root.endsWith('/') ? root : `${root}/`;
    };

    /* --- строка состояния внизу экрана ----------------------------------- */
    const toast = document.createElement('div');
    toast.className = 'fq-import-toast';
    toast.setAttribute('role', 'status');
    document.body.appendChild(toast);
    let hideTimer = null;
    const show = (message, error = false, ms = 0) => {
        clearTimeout(hideTimer);
        toast.textContent = message;
        toast.dataset.error = error ? 'true' : 'false';
        toast.dataset.visible = 'true';
        if (ms) hideTimer = setTimeout(() => { toast.dataset.visible = 'false'; }, ms);
    };
    const hide = () => { toast.dataset.visible = 'false'; };

    /* --- перевод ------------------------------------------------------------ */
    const rasterizeSvg = file => new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file);
        const image = new Image();
        image.onload = () => {
            const width = image.naturalWidth || SVG_SIDE;
            const height = image.naturalHeight || SVG_SIDE;
            const scale = SVG_SIDE / Math.max(width, height);
            const canvas = document.createElement('canvas');
            canvas.width = Math.max(1, Math.round(width * scale));
            canvas.height = Math.max(1, Math.round(height * scale));
            canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(url);
            canvas.toBlob(blob => (blob ? resolve(blob) : reject(new Error('SVG rasterization failed'))), 'image/png');
        };
        image.onerror = () => { URL.revokeObjectURL(url); reject(new Error('not a readable SVG')); };
        image.src = url;
    });

    const viaServer = async file => {
        const response = await fetch(`${rootUrl()}${PATH}`, {
            method: 'POST',
            body: file,
            credentials: 'same-origin',
            headers: { 'X-File-Name': encodeURIComponent(file.name || '') },
        });
        if (!response.ok) {
            let message = `HTTP ${response.status}`;
            try { message = (await response.json()).error || message; } catch { /* не JSON */ }
            throw new Error(message);
        }
        return response.blob();
    };

    const convert = async file => {
        const svg = extension(file.name) === '.svg' || file.type === 'image/svg+xml';
        const blob = svg ? await rasterizeSvg(file) : await viaServer(file);
        return new File([blob], `${stem(file.name)}.png`, { type: 'image/png', lastModified: file.lastModified });
    };

    // Файлы, которым перевод не нужен, — как есть; неудачные выпадают с
    // сообщением, остальные идут дальше.
    const normalize = async files => {
        const result = [];
        for (const file of files) {
            if (!needs(file)) { result.push(file); continue; }
            show(say('import_converting', { name: file.name }));
            try {
                result.push(await convert(file));
                hide();
            } catch (error) {
                show(say('import_failed', { name: file.name, error: error.message }), true, 8000);
            }
        }
        return result;
    };
    const asTransfer = files => {
        const transfer = new DataTransfer();
        files.forEach(file => transfer.items.add(file));
        return transfer;
    };

    /* --- перехват ------------------------------------------------------------- */
    const replayed = new WeakSet();

    window.addEventListener('drop', event => {
        if (replayed.has(event) || !event.dataTransfer) return;
        const files = [...event.dataTransfer.files];
        if (!files.some(needs)) return;
        // Настоящая цель — и внутри теневого DOM кисти правки.
        const target = event.composedPath()[0];
        event.preventDefault();
        event.stopImmediatePropagation();
        normalize(files).then(ready => {
            if (!ready.length) return;
            const replay = new DragEvent('drop', {
                bubbles: true, cancelable: true, composed: true,
                clientX: event.clientX, clientY: event.clientY, dataTransfer: asTransfer(ready),
            });
            replayed.add(replay);
            target.dispatchEvent(replay);
        });
    }, true);

    window.addEventListener('change', event => {
        const input = event.target;
        if (replayed.has(event) || !(input instanceof HTMLInputElement) || input.type !== 'file') return;
        const files = [...(input.files || [])];
        if (!files.some(needs)) return;
        event.stopImmediatePropagation();
        normalize(files).then(ready => {
            if (!ready.length) { input.value = ''; return; }
            input.files = asTransfer(ready).files;
            const replay = new Event('change', { bubbles: true });
            replayed.add(replay);
            input.dispatchEvent(replay);
        });
    }, true);

    /* --- выбор файла: показать и эти форматы ---------------------------------
       ``accept="image/*"`` в диалоге Windows — список расширений, которые
       система считает картинками; HEIC и RAW туда обычно не входят. */
    const ACCEPT = ['image/*', ...CONVERTED].join(',');
    const widen = root => {
        const inputs = root.querySelectorAll ? root.querySelectorAll('input[type="file"]') : [];
        inputs.forEach(input => {
            const accept = input.getAttribute('accept') || '';
            if (accept.split(',').map(s => s.trim()).includes('image/*') && accept !== ACCEPT) {
                input.setAttribute('accept', ACCEPT);
            }
        });
    };
    widen(document);
    // Родитель — чтобы попало и само добавленное поле; своя же правка
    // атрибута повторно ничего не меняет (accept уже равен ACCEPT).
    new MutationObserver(records => {
        for (const record of records) {
            const nodes = record.type === 'attributes' ? [record.target] : [...record.addedNodes];
            nodes.forEach(node => { if (node.nodeType === 1) widen(node.parentNode || node); });
        }
    }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['accept'] });

    // Для кисти правки: её поле выбора файла — в теневом DOM, куда ни
    // перехват ``change``, ни наблюдатель не достают.
    window.__fqImport = { needs, normalize, accept: ACCEPT };
    return [];
}
