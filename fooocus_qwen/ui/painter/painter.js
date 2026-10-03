/* Кисть маски — клиентская часть компонента MaskPainter.

   Исполняется Gradio как тело функции js_on_load с аргументами
   `element` (узел компонента), `trigger` и `props` (живые свойства).
   Перед исполнением Python дописывает сверху константу QP_CSS со стилями.

   Зачем своя кисть вместо gr.ImageEditor. Его кисть на каждом движении мыши
   перерисовывает весь накопленный мазок в текстуру размером с изображение,
   умноженное на плотность пикселей экрана: стоимость движения растёт с
   длиной мазка, и на кадре 3840×2160 при плотности 2x она за один мазок
   выросла с 16.7 до 44 мс — кисть заметно отстаёт от руки. Здесь каждое
   движение рисует только новый отрезок поверх уже нарисованного: стоимость
   постоянна при любой длине мазка и любом размере кадра.

   Разметка живёт в теневом DOM. Gradio при каждом изменении свойств
   переписывает содержимое компонента по шаблону; в светлом DOM от холста
   ничего бы не осталось, в теневой корень он не заглядывает. Шаблон в
   светлом DOM служит только каналом «сервер → кисть»: его перерисовку
   ловит MutationObserver, а свежие значения берутся из `props`. */

const win = element.ownerDocument.defaultView;
const doc = element.ownerDocument;

if (element.__qsPainter) element.__qsPainter.destroy();
const shadow = element.shadowRoot || element.attachShadow({ mode: 'open' });

const LABELS = props.labels || {};
const PALETTE = Array.isArray(props.palette) && props.palette.length ? props.palette : ['#ff0000'];
// Свой цвет и непрозрачность кисти — для эскиза. В маске их нет намеренно:
// маска бинаризуется по альфе 128, и полупрозрачный мазок молча пропадал бы
// из неё или попадал целиком — в зависимости от непрозрачности.
const FREE_COLOUR = Boolean(props.free_colour);
const MASK_COLOR = '#ff2d55';
const MIN_SIZE = 1;
const MAX_SIZE = 600;
const SYNC_DELAY_MS = 400;

let lang = props.lang === 'en' ? 'en' : 'ru';
const say = key => {
    const pair = LABELS[key];
    return pair ? pair[lang === 'en' ? 1 : 0] : key;
};

/* --- разметка ------------------------------------------------------------ */

const ICONS = {
    brush: 'M7 14c-1.66 0-3 1.34-3 3 0 1.31-1.16 2-2 2 .92 1.22 2.49 2 4 2 2.21 0 4-1.79 4-4 0-1.66-1.34-3-3-3zm13.71-9.37-1.34-1.34a1 1 0 0 0-1.41 0L9 12.25 11.75 15l8.96-8.96a1 1 0 0 0 0-1.41z',
    eraser: 'M16.24 3.56 21.19 8.5c.78.79.78 2.05 0 2.84L12 20.53a4.01 4.01 0 0 1-5.66 0L2.81 17c-.78-.79-.78-2.05 0-2.84l10.6-10.6c.79-.78 2.05-.78 2.83 0M4.22 15.58l3.54 3.53c.78.79 2.04.79 2.83 0l3.53-3.53-4.95-4.95-4.95 4.95z',
    undo: 'M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 11.03 17.15 8 12.5 8z',
    redo: 'M18.4 10.6C16.55 8.99 14.15 8 11.5 8c-4.65 0-8.58 3.03-9.96 7.22L3.9 16c1.05-3.19 4.05-5.5 7.6-5.5 1.95 0 3.73.72 5.12 1.88L13 16h9V7l-3.6 3.6z',
    clear: 'M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z',
    invert: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm1 17.93V4.07A8 8 0 0 1 13 19.93z',
    eye: 'M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17a5 5 0 1 1 0-10 5 5 0 0 1 0 10zm0-8a3 3 0 1 0 0 6 3 3 0 0 0 0-6z',
    fit: 'M3 5v4h2V5h4V3H5c-1.1 0-2 .9-2 2zm2 10H3v4c0 1.1.9 2 2 2h4v-2H5v-4zm14 4h-4v2h4c1.1 0 2-.9 2-2v-4h-2v4zm0-16h-4v2h4v4h2V5c0-1.1-.9-2-2-2z',
    open: 'M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z',
    remove: 'M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z',
    crop: 'M17 15h2V7c0-1.1-.9-2-2-2H9v2h8v8zM7 17V1H5v4H1v2h4v10c0 1.1.9 2 2 2h10v4h2v-4h4v-2H7z',
    check: 'M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z',
};
// Пропорции рамки кадрирования: свободно, как у исходника, и соотношения
// карточки модели — те же, что в «Соотношении сторон» и расширении кадра.
const CROP_RATIOS = ['', 'original', '1:1', '4:3', '3:4', '3:2', '2:3', '16:9', '9:16'];
const CROP_HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];
const MIN_CROP = 32;
const icon = name => `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${ICONS[name]}"/></svg>`;

shadow.innerHTML = `
<style>${typeof QP_CSS === 'string' ? QP_CSS : ''}</style>
<div class="qp" data-state="empty">
  <div class="qp-toolbar" role="toolbar">
    <div class="qp-group" data-needs-image>
      <button class="qp-btn" data-act="brush" data-tip="painter_brush" aria-pressed="true">${icon('brush')}</button>
      <button class="qp-btn" data-act="eraser" data-tip="painter_eraser" aria-pressed="false">${icon('eraser')}</button>
    </div>
    <label class="qp-size" data-needs-image>
      <span class="qp-size-label" data-text="painter_size"></span>
      <input type="range" class="qp-size-range" min="${MIN_SIZE}" max="${MAX_SIZE}" step="1">
      <output class="qp-size-value"></output>
    </label>
    <div class="qp-group qp-palette" data-needs-image></div>
    <label class="qp-colour" data-needs-image data-tip="painter_colour" hidden>
      <input type="color" class="qp-colour-input" value="#000000">
    </label>
    <label class="qp-size qp-alpha" data-needs-image hidden>
      <span class="qp-size-label" data-text="painter_opacity"></span>
      <input type="range" class="qp-alpha-range" min="5" max="100" step="5" value="100">
      <output class="qp-size-value qp-alpha-value"></output>
    </label>
    <div class="qp-group" data-needs-image>
      <button class="qp-btn" data-act="undo" data-tip="painter_undo">${icon('undo')}</button>
      <button class="qp-btn" data-act="redo" data-tip="painter_redo">${icon('redo')}</button>
      <button class="qp-btn" data-act="invert" data-tip="painter_invert">${icon('invert')}</button>
      <button class="qp-btn" data-act="clear" data-tip="painter_clear">${icon('clear')}</button>
      <button class="qp-btn" data-act="toggle" data-tip="painter_toggle" aria-pressed="true">${icon('eye')}</button>
    </div>
    <div class="qp-group" data-needs-image data-crop-tool>
      <button class="qp-btn" data-act="crop" data-tip="painter_crop" aria-pressed="false">${icon('crop')}</button>
    </div>
    <div class="qp-group qp-cropbar" hidden>
      <select class="qp-crop-ratio"></select>
      <button class="qp-btn" data-act="cropApply" data-tip="painter_crop_apply">${icon('check')}</button>
      <button class="qp-btn" data-act="cropCancel" data-tip="painter_crop_cancel">${icon('remove')}</button>
    </div>
    <div class="qp-spacer"></div>
    <div class="qp-group" data-needs-image>
      <output class="qp-zoom" aria-live="off"></output>
      <button class="qp-btn" data-act="fit" data-tip="painter_fit">${icon('fit')}</button>
    </div>
    <div class="qp-group">
      <button class="qp-btn" data-act="open" data-tip="painter_open">${icon('open')}</button>
      <button class="qp-btn" data-act="remove" data-tip="painter_remove" data-needs-image>${icon('remove')}</button>
    </div>
  </div>
  <div class="qp-stage" tabindex="0">
    <div class="qp-view">
      <canvas class="qp-image"></canvas>
      <canvas class="qp-layer"></canvas>
      <canvas class="qp-live"></canvas>
    </div>
    <div class="qp-cursor" hidden></div>
    <div class="qp-crop" hidden>
      <div class="qp-crop-box">${CROP_HANDLES.map(h => `<span class="qp-crop-handle" data-h="${h}"></span>`).join('')}</div>
    </div>
    <button class="qp-empty" data-act="open">
      <span class="qp-empty-title" data-text="painter_empty_title"></span>
      <span class="qp-empty-hint" data-text="painter_empty_hint"></span>
    </button>
    <div class="qp-notice" hidden></div>
    <div class="qp-busy" hidden><span data-text="painter_busy"></span></div>
  </div>
  <div class="qp-status">
    <span class="qp-hint" data-text="painter_hint"></span>
    <span class="qp-message" role="status"></span>
  </div>
  <input class="qp-file" type="file" accept="image/*" hidden>
</div>`;

const $ = selector => shadow.querySelector(selector);
const ui = {
    root: $('.qp'), stage: $('.qp-stage'), view: $('.qp-view'),
    image: $('.qp-image'), layer: $('.qp-layer'), cursor: $('.qp-cursor'),
    range: $('.qp-size-range'), sizeValue: $('.qp-size-value'), zoom: $('.qp-zoom'),
    palette: $('.qp-palette'), notice: $('.qp-notice'), busy: $('.qp-busy'),
    message: $('.qp-message'), file: $('.qp-file'), live: $('.qp-live'),
    colour: $('.qp-colour'), colourInput: $('.qp-colour-input'),
    alpha: $('.qp-alpha'), alphaRange: $('.qp-alpha-range'), alphaValue: $('.qp-alpha-value'),
    crop: $('.qp-crop'), cropBox: $('.qp-crop-box'), cropBar: $('.qp-cropbar'), cropRatio: $('.qp-crop-ratio'),
    hint: $('.qp-hint'),
};
const imageCtx = ui.image.getContext('2d');
const layerCtx = ui.layer.getContext('2d');
const liveCtx = ui.live.getContext('2d');

/* --- состояние ----------------------------------------------------------- */

const state = {
    region: props.region || 'mask',
    tool: 'brush',
    color: PALETTE[0],
    alpha: 1,            // непрозрачность кисти (только при FREE_COLOUR)
    size: 24,
    width: 0,
    height: 0,
    sourcePath: null,
    pendingUpload: null,
    base: null,          // слой, пришедший с сервера: точка отсчёта отмены
    history: [],         // команды: {kind: 'stroke'|'clear'|'invert', ...}
    future: [],
    visible: true,
    view: { s: 1, x: 0, y: 0, fit: true },
    stroke: null,
    pan: null,
    spaceDown: false,
    hovered: false,
    lastServerRev: null,
    lastClientValue: null,
    syncTimer: null,
    touches: new Map(),
    pinch: null,
    crop: null,          // рамка кадрирования {x, y, w, h} в пикселях исходника
    cropRatio: '',
    cropDrag: null,
    previous: [],        // снимки до кадрирования: Ctrl+Z возвращает исходник
};

/* --- тексты и режим ------------------------------------------------------ */

function relabel() {
    shadow.querySelectorAll('[data-text]').forEach(node => { node.textContent = say(node.dataset.text); });
    shadow.querySelectorAll('[data-tip]').forEach(node => {
        const text = say(node.dataset.tip);
        node.title = text;
        node.setAttribute('aria-label', text);
    });
    ui.range.setAttribute('aria-label', say('painter_size'));
    ui.stage.setAttribute('aria-label', say('painter_stage'));
    ui.cropRatio.innerHTML = '';
    CROP_RATIOS.forEach(key => {
        const option = doc.createElement('option');
        option.value = key;
        option.textContent = key === '' ? say('painter_crop_free') : key === 'original' ? say('painter_crop_original') : key;
        ui.cropRatio.appendChild(option);
    });
    ui.cropRatio.value = state.cropRatio;
    ui.cropRatio.title = say('painter_crop_ratio');
    ui.cropRatio.setAttribute('aria-label', say('painter_crop_ratio'));
    ui.hint.textContent = say(state.crop ? 'painter_crop_hint' : 'painter_hint');
    updateNotice();
}

function buildPalette() {
    ui.palette.innerHTML = '';
    PALETTE.forEach(color => {
        const swatch = doc.createElement('button');
        swatch.className = 'qp-swatch';
        swatch.style.setProperty('--swatch', color);
        swatch.dataset.color = color;
        swatch.setAttribute('aria-label', color);
        swatch.addEventListener('click', () => { state.color = color; state.tool = 'brush'; refreshTools(); });
        ui.palette.appendChild(swatch);
    });
}

function paintColor() {
    return state.region === 'annotation' ? state.color : MASK_COLOR;
}

function applyRegion(region) {
    state.region = region || 'mask';
    ui.root.dataset.region = state.region;
    ui.layer.style.opacity = state.visible ? (state.region === 'annotation' ? '1' : '0.55') : '0';
    updateNotice();
    refreshTools();
}

function updateNotice() {
    const text = state.region === 'none' && state.width ? say('painter_region_none') : '';
    ui.notice.textContent = text;
    ui.notice.hidden = !text;
}

function refreshTools() {
    shadow.querySelector('[data-act="brush"]').setAttribute('aria-pressed', String(state.tool === 'brush'));
    shadow.querySelector('[data-act="eraser"]').setAttribute('aria-pressed', String(state.tool === 'eraser'));
    shadow.querySelector('[data-act="toggle"]').setAttribute('aria-pressed', String(state.visible));
    shadow.querySelector('[data-act="undo"]').disabled = !state.history.length && !state.previous.length;
    shadow.querySelector('[data-act="crop"]').setAttribute('aria-pressed', String(Boolean(state.crop)));
    shadow.querySelector('[data-act="redo"]').disabled = !state.future.length;
    ui.palette.hidden = state.region !== 'annotation';
    ui.palette.querySelectorAll('.qp-swatch').forEach(node => {
        node.setAttribute('aria-pressed', String(node.dataset.color === state.color));
    });
    const free = FREE_COLOUR && state.region === 'annotation';
    ui.colour.hidden = ui.alpha.hidden = !free;
    // Эскиз: «инвертировать разметку» — инструмент маски, на рисунке он залил бы
    // холст цветом; место на панели нужнее своему цвету и непрозрачности.
    ui.root.dataset.free = String(free);
    shadow.querySelector('[data-act="invert"]').hidden = free;
    // Кадрирование — для исходника правки; холсту эскиза оно ни к чему.
    shadow.querySelector('[data-crop-tool]').hidden = FREE_COLOUR;
    // Свой цвет «нажат», когда выбранный цвет не из палитры.
    ui.colour.setAttribute('aria-pressed', String(free && !PALETTE.includes(state.color)));
    if (ui.colourInput.value !== state.color) ui.colourInput.value = state.color;
    ui.alphaRange.value = String(Math.round(state.alpha * 100));
    ui.alphaValue.textContent = `${Math.round(state.alpha * 100)}%`;
    ui.range.value = String(state.size);
    ui.sizeValue.textContent = `${state.size}px`;
    ui.zoom.textContent = state.width ? `${Math.round(state.view.s * 100)}%` : '';
    ui.stage.dataset.tool = state.spaceDown || state.pan ? 'pan' : state.tool;
    updateCursor();
}

function flash(text, isError = false) {
    ui.message.textContent = text;
    ui.message.dataset.error = String(isError);
    clearTimeout(flash.timer);
    flash.timer = win.setTimeout(() => { ui.message.textContent = ''; }, isError ? 6000 : 2500);
}

function setBusy(busy) { ui.busy.hidden = !busy; }

/* --- вид: масштаб и сдвиг ------------------------------------------------ */

const PAD = 12;

function stageBox() { return ui.stage.getBoundingClientRect(); }

function fitScale() {
    const box = stageBox();
    if (!state.width || !box.width || !box.height) return 1;
    return Math.min((box.width - PAD * 2) / state.width, (box.height - PAD * 2) / state.height);
}

function fit() {
    const box = stageBox();
    const s = fitScale();
    state.view = { s, x: (box.width - state.width * s) / 2, y: (box.height - state.height * s) / 2, fit: true };
    applyView();
}

function applyView() {
    const { s, x, y } = state.view;
    ui.view.style.width = `${state.width}px`;
    ui.view.style.height = `${state.height}px`;
    ui.view.style.transform = `translate(${x}px, ${y}px) scale(${s})`;
    // Крупный масштаб показывает пиксели как есть: маску ведут по краю объекта,
    // и размытое увеличение этому мешает.
    ui.view.dataset.pixelated = String(s >= 3);
    refreshTools();
    showCrop();
}

function zoomAt(factor, clientX, clientY) {
    if (!state.width) return;
    const box = stageBox();
    const px = clientX - box.left, py = clientY - box.top;
    const minScale = Math.min(fitScale(), 1) * 0.5;
    const s = Math.min(Math.max(state.view.s * factor, minScale), 32);
    const k = s / state.view.s;
    state.view = { s, x: px - (px - state.view.x) * k, y: py - (py - state.view.y) * k, fit: false };
    applyView();
}

function toImage(event) {
    const box = stageBox();
    return {
        x: (event.clientX - box.left - state.view.x) / state.view.s,
        y: (event.clientY - box.top - state.view.y) / state.view.s,
    };
}

function updateCursor(event) {
    if (event) state.pointer = { x: event.clientX, y: event.clientY };
    const drawing = state.width && state.hovered && !state.spaceDown && !state.pan && state.pointer && !state.crop;
    ui.cursor.hidden = !drawing;
    if (!drawing) return;
    const box = stageBox();
    const d = Math.max(state.size * 2 * state.view.s, 4);
    ui.cursor.style.width = ui.cursor.style.height = `${d}px`;
    ui.cursor.style.transform = `translate(${state.pointer.x - box.left - d / 2}px, ${state.pointer.y - box.top - d / 2}px)`;
    ui.cursor.dataset.tool = state.tool;
}

/* --- рисование ------------------------------------------------------------- */

function segment(ctx, stroke, a, b) {
    ctx.save();
    ctx.globalCompositeOperation = stroke.erase ? 'destination-out' : 'source-over';
    ctx.strokeStyle = ctx.fillStyle = stroke.color;
    ctx.lineWidth = stroke.size * 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (a.x === b.x && a.y === b.y) {
        ctx.beginPath();
        ctx.arc(a.x, a.y, stroke.size, 0, Math.PI * 2);
        ctx.fill();
    } else {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
    }
    ctx.restore();
}

function replayStroke(ctx, stroke) {
    const p = stroke.points;
    // Мазок перерисовывается одним контуром, поэтому полупрозрачный ложится
    // ровно, без потемнений на стыках отрезков.
    const alpha = stroke.alpha === undefined ? 1 : stroke.alpha;
    if (p.length === 2) {
        ctx.save();
        ctx.globalAlpha = alpha;
        segment(ctx, stroke, { x: p[0], y: p[1] }, { x: p[0], y: p[1] });
        ctx.restore();
        return;
    }
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.globalCompositeOperation = stroke.erase ? 'destination-out' : 'source-over';
    ctx.strokeStyle = stroke.color;
    ctx.lineWidth = stroke.size * 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(p[0], p[1]);
    for (let i = 2; i < p.length; i += 2) ctx.lineTo(p[i], p[i + 1]);
    ctx.stroke();
    ctx.restore();
}

function invertLayer(ctx, color) {
    const copy = doc.createElement('canvas');
    copy.width = state.width;
    copy.height = state.height;
    copy.getContext('2d').drawImage(ui.layer, 0, 0);
    ctx.save();
    ctx.clearRect(0, 0, state.width, state.height);
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, state.width, state.height);
    ctx.globalCompositeOperation = 'destination-out';
    ctx.drawImage(copy, 0, 0);
    ctx.restore();
}

function run(command, ctx = layerCtx) {
    if (command.kind === 'stroke') replayStroke(ctx, command);
    else if (command.kind === 'clear') ctx.clearRect(0, 0, state.width, state.height);
    else if (command.kind === 'invert') invertLayer(ctx, command.color);
}

function redraw() {
    layerCtx.clearRect(0, 0, state.width, state.height);
    if (state.base) layerCtx.drawImage(state.base, 0, 0, state.width, state.height);
    state.history.forEach(command => run(command));
}

function commit(command) {
    state.history.push(command);
    state.future = [];
    refreshTools();
    scheduleSync();
}

/* Полупрозрачный мазок. Каждое движение рисует свой отрезок, и отрезки
   перекрываются круглыми концами: нарисованные прямо в слой с альфой, они
   темнели бы на каждом стыке, и мазок выходил бы пятнистым. Поэтому такой
   мазок рисуется непрозрачно на отдельном живом холсте, который на экране
   показан с нужной непрозрачностью, а на отпускании сливается в слой одним
   drawImage с globalAlpha. Стоимость движения — прежняя, один отрезок. */
function strokeTarget(stroke) {
    return stroke.alpha < 1 ? liveCtx : layerCtx;
}

function beginStroke(event, erase) {
    const p = toImage(event);
    const alpha = FREE_COLOUR && state.region === 'annotation' && !erase ? state.alpha : 1;
    state.stroke = { kind: 'stroke', erase, color: paintColor(), size: state.size, alpha, points: [p.x, p.y], last: p };
    if (alpha < 1) {
        liveCtx.clearRect(0, 0, state.width, state.height);
        ui.live.style.opacity = String(alpha);
    }
    segment(strokeTarget(state.stroke), state.stroke, p, p);
}

function extendStroke(event) {
    const stroke = state.stroke;
    const target = strokeTarget(stroke);
    const events = typeof event.getCoalescedEvents === 'function' ? event.getCoalescedEvents() : [];
    for (const item of events.length ? events : [event]) {
        const p = toImage(item);
        if (Math.abs(p.x - stroke.last.x) < 0.5 && Math.abs(p.y - stroke.last.y) < 0.5) continue;
        segment(target, stroke, stroke.last, p);
        stroke.points.push(p.x, p.y);
        stroke.last = p;
    }
}

function endStroke() {
    const stroke = state.stroke;
    state.stroke = null;
    if (!stroke) return;
    delete stroke.last;
    if (stroke.alpha < 1) {
        layerCtx.save();
        layerCtx.globalAlpha = stroke.alpha;
        layerCtx.drawImage(ui.live, 0, 0);
        layerCtx.restore();
        liveCtx.clearRect(0, 0, state.width, state.height);
    }
    commit(stroke);
}

function cancelStroke() {
    if (!state.stroke) return;
    state.stroke = null;
    liveCtx.clearRect(0, 0, state.width, state.height);
    redraw();
}

function undo() {
    if (!state.history.length && state.previous.length) { restoreSnapshot(state.previous.pop()); return; }
    if (!state.history.length) return;
    state.future.push(state.history.pop());
    redraw();
    refreshTools();
    scheduleSync();
}

function redo() {
    if (!state.future.length) return;
    const command = state.future.pop();
    state.history.push(command);
    run(command);
    refreshTools();
    scheduleSync();
}

/* --- указатель ------------------------------------------------------------- */

function onPointerDown(event) {
    if (!state.width) return;
    ui.stage.focus({ preventScroll: true });
    if (event.pointerType === 'touch') {
        state.touches.set(event.pointerId, { x: event.clientX, y: event.clientY });
        if (state.touches.size === 2) { cancelStroke(); startPinch(); return; }
    }
    const panning = event.button === 1 || state.spaceDown;
    if (panning) {
        state.pan = { x: event.clientX, y: event.clientY, vx: state.view.x, vy: state.view.y };
    } else if (state.crop) {
        if (event.button !== 0) return;
        cropPointerDown(event);
    } else if (event.button === 0 || event.button === 2) {
        beginStroke(event, event.button === 2 || state.tool === 'eraser');
    } else {
        return;
    }
    ui.stage.setPointerCapture(event.pointerId);
    event.preventDefault();
    refreshTools();
}

function onPointerMove(event) {
    if (event.pointerType === 'touch' && state.touches.has(event.pointerId)) {
        state.touches.set(event.pointerId, { x: event.clientX, y: event.clientY });
        if (state.pinch) { movePinch(); return; }
    }
    if (state.pan) {
        state.view = { ...state.view, x: state.pan.vx + event.clientX - state.pan.x,
                       y: state.pan.vy + event.clientY - state.pan.y, fit: false };
        applyView();
    } else if (state.cropDrag) {
        cropPointerMove(event);
    } else if (state.stroke) {
        extendStroke(event);
    }
    updateCursor(event);
}

function onPointerUp(event) {
    state.touches.delete(event.pointerId);
    if (state.pinch && state.touches.size < 2) state.pinch = null;
    if (state.pan) state.pan = null;
    state.cropDrag = null;
    if (state.stroke) endStroke();
    if (ui.stage.hasPointerCapture(event.pointerId)) ui.stage.releasePointerCapture(event.pointerId);
    refreshTools();
}

function startPinch() {
    const [a, b] = [...state.touches.values()];
    state.pinch = { distance: Math.hypot(a.x - b.x, a.y - b.y), mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } };
}

function movePinch() {
    const [a, b] = [...state.touches.values()];
    const distance = Math.hypot(a.x - b.x, a.y - b.y);
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    state.view = { ...state.view, x: state.view.x + mid.x - state.pinch.mid.x, y: state.view.y + mid.y - state.pinch.mid.y, fit: false };
    zoomAt(distance / (state.pinch.distance || distance), mid.x, mid.y);
    state.pinch = { distance, mid };
}

function onWheel(event) {
    if (!state.width) return;
    event.preventDefault();
    const delta = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY;
    if (event.shiftKey) {
        setSize(state.size * Math.exp(-delta * 0.002));
    } else {
        zoomAt(Math.exp(-delta * 0.0015), event.clientX, event.clientY);
    }
}

function setSize(value) {
    state.size = Math.round(Math.min(Math.max(value, MIN_SIZE), MAX_SIZE));
    refreshTools();
}

/* --- клавиатура ---------------------------------------------------------- */

function typingInto(event) {
    const target = event.composedPath()[0];
    return target instanceof win.HTMLElement && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))
        && !shadow.contains(target);
}

function ownsKeyboard() {
    return state.width && (state.hovered || shadow.activeElement);
}

function onKeyDown(event) {
    if (!ownsKeyboard() || typingInto(event)) return;
    const ctrl = event.ctrlKey || event.metaKey;
    const key = event.key.toLowerCase();
    let handled = true;
    if (state.crop && key === 'enter') applyCrop();
    else if (state.crop && key === 'escape') cancelCrop();
    else if (ctrl && key === 'z' && !event.shiftKey) undo();
    else if (ctrl && (key === 'y' || (key === 'z' && event.shiftKey))) redo();
    else if (ctrl) handled = false;
    else if (event.code === 'Space') { if (!state.spaceDown) { state.spaceDown = true; refreshTools(); } }
    else if (key === 'b') { cancelCrop(); state.tool = 'brush'; refreshTools(); }
    else if (key === 'e') { cancelCrop(); state.tool = 'eraser'; refreshTools(); }
    else if (key === 'c' && !FREE_COLOUR) { if (state.crop) cancelCrop(); else startCrop(); }
    else if (key === 'x') { state.tool = state.tool === 'brush' ? 'eraser' : 'brush'; refreshTools(); }
    else if (key === '[') setSize(state.size / 1.15);
    else if (key === ']') setSize(state.size * 1.15);
    else if (key === 'h') toggleVisible();
    else if (key === 'f' || key === '0') fit();
    else handled = false;
    if (handled) { event.preventDefault(); event.stopPropagation(); }
}

function onKeyUp(event) {
    if (event.code === 'Space' && state.spaceDown) { state.spaceDown = false; refreshTools(); }
}

function toggleVisible() {
    state.visible = !state.visible;
    applyRegion(state.region);
}

/* --- изображение: загрузка и снятие -------------------------------------- */

function rootUrl() {
    const root = (win.gradio_config && win.gradio_config.root) || `${win.location.origin}${win.location.pathname}`;
    return root.endsWith('/') ? root : `${root}/`;
}

function fileUrl(path) {
    return `${rootUrl()}gradio_api/file=${encodeURI(path)}`;
}

async function fetchBitmap(url) {
    const response = await win.fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return win.createImageBitmap(await response.blob(), { imageOrientation: 'from-image' });
}

function installImage(bitmap, base = null) {
    state.width = bitmap.width;
    state.height = bitmap.height;
    for (const canvas of [ui.image, ui.layer, ui.live]) {
        canvas.width = state.width;
        canvas.height = state.height;
    }
    imageCtx.drawImage(bitmap, 0, 0);
    state.base = base;
    state.history = [];
    state.future = [];
    state.size = Math.max(4, Math.round(Math.min(state.width, state.height) / 40));
    ui.root.dataset.state = 'ready';
    redraw();
    fit();
    applyRegion(state.region);
}

async function upload(file) {
    const form = new win.FormData();
    form.append('files', file, file.name || 'image.png');
    const response = await win.fetch(`${rootUrl()}gradio_api/upload`, { method: 'POST', body: form });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const paths = await response.json();
    if (!Array.isArray(paths) || !paths[0]) throw new Error('empty upload response');
    return paths[0];
}

async function loadFile(file) {
    if (!file || !/^image\//.test(file.type)) { flash(say('painter_not_image'), true); return; }
    setBusy(true);
    try {
        const bitmap = await win.createImageBitmap(file, { imageOrientation: 'from-image' });
        state.sourcePath = null;
        state.lastServerRev = null;
        state.previous = [];
        cancelCrop();
        installImage(bitmap);
        // Картинка уже на экране; загрузка на сервер идёт следом и не держит
        // человека: рисовать можно сразу, а «Применить» дождётся её сам.
        state.pendingUpload = upload(file)
            .then(path => { state.sourcePath = path; scheduleSync(0); })
            .catch(error => { flash(`${say('painter_upload_failed')}: ${error.message}`, true); })
            .finally(() => { state.pendingUpload = null; });
    } catch (error) {
        flash(`${say('painter_not_image')}: ${error.message}`, true);
    } finally {
        setBusy(false);
    }
}

async function loadServerValue(data) {
    setBusy(true);
    try {
        const bitmap = await fetchBitmap(fileUrl(data.source));
        const base = data.layer ? await fetchBitmap(fileUrl(data.layer)) : null;
        state.sourcePath = data.source;
        state.lastServerRev = data.rev;
        state.previous = [];
        cancelCrop();
        installImage(bitmap, base);
        // Слой с сервера (новая площадь при расширении холста) — уже
        // содержательная разметка: её надо отправить обратно как есть.
        scheduleSync(0);
    } catch (error) {
        flash(`${say('painter_load_failed')}: ${error.message}`, true);
    } finally {
        setBusy(false);
    }
}

function removeImage() {
    cancelCrop();
    state.previous = [];
    state.width = state.height = 0;
    state.sourcePath = null;
    state.base = null;
    state.history = [];
    state.future = [];
    state.lastServerRev = null;
    ui.root.dataset.state = 'empty';
    updateNotice();
    refreshTools();
    publish('');
}

/* --- кадрирование ---------------------------------------------------------- */

/* Рамка живёт в координатах исходника, а рисуется поверх холста в экранных:
   ручки одного размера при любом масштабе. Обрезка — по пикселям исходника,
   без пересжатия; разметка обрезается вместе с ним. Обрезанная картинка
   уходит на сервер новым исходником, а снимок прежнего остаётся для Ctrl+Z. */

const clamp = (value, low, high) => Math.min(Math.max(value, low), high);

function ratioValue(key) {
    if (!key) return 0;
    if (key === 'original') return state.width / state.height;
    const [a, b] = key.split(':').map(Number);
    return a / b;
}

function clampRect(rect) {
    const w = clamp(rect.w, Math.min(MIN_CROP, state.width), state.width);
    const h = clamp(rect.h, Math.min(MIN_CROP, state.height), state.height);
    return { x: clamp(rect.x, 0, state.width - w), y: clamp(rect.y, 0, state.height - h), w, h };
}

/* Наибольшая рамка заданной пропорции в пределах кадра — по центру прежней. */
function fitRatio(rect, ratio) {
    let w = state.width, h = w / ratio;
    if (h > state.height) { h = state.height; w = h * ratio; }
    const cx = rect.x + rect.w / 2, cy = rect.y + rect.h / 2;
    return clampRect({ x: cx - w / 2, y: cy - h / 2, w, h });
}

function startCrop() {
    if (!state.width || FREE_COLOUR) return;
    if (state.stroke) endStroke();
    const whole = { x: 0, y: 0, w: state.width, h: state.height };
    const ratio = ratioValue(state.cropRatio);
    state.crop = ratio ? fitRatio(whole, ratio) : whole;
    showCrop();
    relabel();
    refreshTools();
}

function cancelCrop() {
    if (!state.crop) return;
    state.crop = null;
    state.cropDrag = null;
    showCrop();
    relabel();
    refreshTools();
}

function setCropRatio(key) {
    state.cropRatio = key;
    if (!state.crop) return;
    const ratio = ratioValue(key);
    if (ratio) state.crop = fitRatio(state.crop, ratio);
    showCrop();
}

function showCrop() {
    const rect = state.crop;
    ui.crop.hidden = !rect;
    ui.cropBar.hidden = !rect;
    ui.stage.dataset.cropping = String(Boolean(rect));
    if (!rect) return;
    const { s, x, y } = state.view;
    Object.assign(ui.cropBox.style, {
        left: `${x + rect.x * s}px`, top: `${y + rect.y * s}px`,
        width: `${rect.w * s}px`, height: `${rect.h * s}px`,
    });
    ui.cropBox.dataset.size = `${Math.round(rect.w)}×${Math.round(rect.h)}`;
    ui.crop.dataset.locked = String(Boolean(ratioValue(state.cropRatio)));
    if (ui.cropRatio.value !== state.cropRatio) ui.cropRatio.value = state.cropRatio;
}

function cropPointerDown(event) {
    const p = toImage(event);
    const target = event.composedPath()[0];
    const handle = target && target.dataset ? target.dataset.h : undefined;
    const rect = state.crop;
    const inside = p.x >= rect.x && p.x <= rect.x + rect.w && p.y >= rect.y && p.y <= rect.y + rect.h;
    const start = { x: clamp(p.x, 0, state.width), y: clamp(p.y, 0, state.height) };
    state.cropDrag = { mode: handle ? 'resize' : inside ? 'move' : 'new', handle, start, rect: { ...rect } };
}

function cropPointerMove(event) {
    const drag = state.cropDrag;
    const p = toImage(event);
    if (drag.mode === 'move') {
        state.crop = clampRect({ ...drag.rect, x: drag.rect.x + p.x - drag.start.x, y: drag.rect.y + p.y - drag.start.y });
        showCrop();
        return;
    }
    // Неподвижный угол (или край) и подвижная точка — рамка между ними.
    const r = drag.rect;
    let ax, ay, bx, by;
    if (drag.mode === 'new') {
        ax = drag.start.x; ay = drag.start.y; bx = p.x; by = p.y;
    } else {
        const h = drag.handle;
        const horizontal = h.includes('w') || h.includes('e');
        const vertical = h.includes('n') || h.includes('s');
        ax = h.includes('w') ? r.x + r.w : r.x;
        bx = horizontal ? p.x : r.x + r.w;
        ay = h.includes('n') ? r.y + r.h : r.y;
        by = vertical ? p.y : r.y + r.h;
    }
    bx = clamp(bx, 0, state.width);
    by = clamp(by, 0, state.height);
    let w = Math.abs(bx - ax), h = Math.abs(by - ay);
    const ratio = ratioValue(state.cropRatio);
    if (ratio) {
        if (w / Math.max(h, 1e-6) > ratio) w = h * ratio; else h = w / ratio;
        const maxW = bx >= ax ? state.width - ax : ax;
        const maxH = by >= ay ? state.height - ay : ay;
        if (w > maxW) { w = maxW; h = w / ratio; }
        if (h > maxH) { h = maxH; w = h * ratio; }
    }
    w = Math.max(w, Math.min(MIN_CROP, state.width));
    h = Math.max(h, Math.min(MIN_CROP, state.height));
    state.crop = clampRect({ x: bx >= ax ? ax : ax - w, y: by >= ay ? ay : ay - h, w, h });
    showCrop();
}

function cutCanvas(source, x, y, w, h) {
    const canvas = doc.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    canvas.getContext('2d').drawImage(source, x, y, w, h, 0, 0, w, h);
    return canvas;
}

function snapshot() {
    return {
        image: cutCanvas(ui.image, 0, 0, state.width, state.height),
        base: hasMarks() ? cutCanvas(ui.layer, 0, 0, state.width, state.height) : null,
        sourcePath: state.sourcePath,
        size: state.size,
    };
}

function restoreSnapshot(saved) {
    cancelCrop();
    installImage(saved.image, saved.base);
    state.sourcePath = saved.sourcePath;
    state.size = saved.size;
    refreshTools();
    scheduleSync(0);
}

async function applyCrop() {
    const rect = state.crop;
    if (!rect) return;
    const x = Math.round(rect.x), y = Math.round(rect.y);
    const w = Math.round(rect.w), h = Math.round(rect.h);
    cancelCrop();
    if (x === 0 && y === 0 && w === state.width && h === state.height) return;
    // Исходник ещё грузится на сервер — дождаться: снимку нужен его путь.
    if (state.pendingUpload) await state.pendingUpload;
    state.previous.push(snapshot());
    const image = cutCanvas(ui.image, x, y, w, h);
    const layer = hasMarks() ? cutCanvas(ui.layer, x, y, w, h) : null;
    const size = state.size;
    installImage(image, layer);
    state.size = size;
    state.sourcePath = null;
    refreshTools();
    flash(say('painter_cropped').replace('{w}', w).replace('{h}', h));
    const blob = await new Promise(resolve => image.toBlob(resolve, 'image/png'));
    state.pendingUpload = upload(new win.File([blob], 'crop.png', { type: 'image/png' }))
        .then(path => { state.sourcePath = path; scheduleSync(0); })
        .catch(error => { flash(`${say('painter_upload_failed')}: ${error.message}`, true); })
        .finally(() => { state.pendingUpload = null; });
}

/* --- значение: кисть → сервер -------------------------------------------- */

function hasMarks() {
    return Boolean(state.base) || state.history.length > 0;
}

function layerDataUrl() {
    return new Promise((resolve, reject) => {
        ui.layer.toBlob(blob => {
            if (!blob) { reject(new Error('toBlob')); return; }
            const reader = new win.FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error);
            reader.readAsDataURL(blob);
        }, 'image/png');
    });
}

async function exportValue() {
    if (state.pendingUpload) await state.pendingUpload;
    if (!state.width || !state.sourcePath) return '';
    const value = {
        v: 1,
        origin: 'client',
        rev: state.lastServerRev,
        source: state.sourcePath,
        layer: hasMarks() ? await layerDataUrl() : null,
        width: state.width,
        height: state.height,
    };
    // Значение попадает в шаблон внутри <script>: «<» экранируется, чтобы
    // никакая строка не закрыла тег раньше времени.
    return JSON.stringify(value).replace(/</g, '\\u003c');
}

function publish(value) {
    state.lastClientValue = value;
    if (props.value !== value) props.value = value;
}

function scheduleSync(delay = SYNC_DELAY_MS) {
    win.clearTimeout(state.syncTimer);
    state.syncTimer = win.setTimeout(async () => {
        try { publish(await exportValue()); } catch (error) { flash(error.message, true); }
    }, delay);
}

async function flush() {
    win.clearTimeout(state.syncTimer);
    if (state.stroke) endStroke();
    const value = await exportValue();
    publish(value);
    return value;
}

/* --- значение: сервер → кисть ---------------------------------------------- */

function syncFromProps() {
    const nextLang = props.lang === 'en' ? 'en' : 'ru';
    if (nextLang !== lang) { lang = nextLang; relabel(); }
    if ((props.region || 'mask') !== state.region) applyRegion(props.region);

    const raw = props.value || '';
    if (raw === state.lastClientValue) return;
    if (!raw) { if (state.width && state.sourcePath) removeImage(); return; }
    let data;
    try { data = JSON.parse(raw); } catch { return; }
    if (data && data.origin === 'server' && data.rev && data.rev !== state.lastServerRev && data.source) {
        loadServerValue(data);
    }
}

/* --- проводка ------------------------------------------------------------ */

const listeners = [];
function listen(target, type, handler, options) {
    target.addEventListener(type, handler, options);
    listeners.push(() => target.removeEventListener(type, handler, options));
}

listen(ui.stage, 'pointerdown', onPointerDown);
listen(ui.stage, 'pointermove', onPointerMove);
listen(ui.stage, 'pointerup', onPointerUp);
listen(ui.stage, 'pointercancel', event => { cancelStroke(); onPointerUp(event); });
listen(ui.stage, 'pointerenter', event => { state.hovered = true; updateCursor(event); });
listen(ui.stage, 'pointerleave', () => { state.hovered = false; updateCursor(); });
listen(ui.stage, 'wheel', onWheel, { passive: false });
listen(ui.stage, 'contextmenu', event => event.preventDefault());
listen(ui.stage, 'dragover', event => { event.preventDefault(); ui.stage.dataset.drop = 'true'; });
listen(ui.stage, 'dragleave', () => { delete ui.stage.dataset.drop; });
listen(ui.stage, 'drop', event => {
    event.preventDefault();
    delete ui.stage.dataset.drop;
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (file) loadFile(file);
});
listen(doc, 'keydown', onKeyDown, true);
listen(doc, 'keyup', onKeyUp, true);
listen(doc, 'paste', event => {
    if (!(state.hovered || shadow.activeElement) || typingInto(event)) return;
    const item = [...(event.clipboardData ? event.clipboardData.items : [])].find(i => i.type.startsWith('image/'));
    if (item) { event.preventDefault(); loadFile(item.getAsFile()); }
});
listen(ui.range, 'input', () => setSize(Number(ui.range.value)));
listen(ui.colourInput, 'input', () => { state.color = ui.colourInput.value; state.tool = 'brush'; refreshTools(); });
listen(ui.alphaRange, 'input', () => { state.alpha = Number(ui.alphaRange.value) / 100; refreshTools(); });
listen(ui.file, 'change', () => { if (ui.file.files[0]) loadFile(ui.file.files[0]); ui.file.value = ''; });
listen(ui.cropRatio, 'change', () => setCropRatio(ui.cropRatio.value));
listen(ui.crop, 'dblclick', () => applyCrop());

const actions = {
    brush: () => { cancelCrop(); state.tool = 'brush'; refreshTools(); },
    eraser: () => { cancelCrop(); state.tool = 'eraser'; refreshTools(); },
    crop: () => { if (state.crop) cancelCrop(); else startCrop(); },
    cropApply: () => applyCrop(),
    cropCancel: () => cancelCrop(),
    undo, redo,
    invert: () => { const command = { kind: 'invert', color: paintColor() }; run(command); commit(command); },
    clear: () => { if (!hasMarks()) return; const command = { kind: 'clear' }; run(command); commit(command); },
    toggle: toggleVisible,
    fit,
    open: () => ui.file.click(),
    remove: removeImage,
};
shadow.querySelectorAll('[data-act]').forEach(node => {
    listen(node, 'click', () => { const act = actions[node.dataset.act]; if (act) act(); });
});

const resizeObserver = new win.ResizeObserver(() => {
    if (!state.width) return;
    if (state.view.fit) fit(); else applyView();
});
resizeObserver.observe(ui.stage);

const mutationObserver = new win.MutationObserver(syncFromProps);
mutationObserver.observe(element, { subtree: true, childList: true, characterData: true, attributes: true });

buildPalette();
relabel();
applyRegion(state.region);
refreshTools();
syncFromProps();

/* --- программный интерфейс -------------------------------------------------- */

const api = {
    export: exportValue,
    flush,
    state: () => ({
        width: state.width, height: state.height, tool: state.tool, size: state.size,
        color: state.color, alpha: state.alpha, freeColour: FREE_COLOUR,
        region: state.region, strokes: state.history.length, source: state.sourcePath,
        scale: state.view.s, uploading: Boolean(state.pendingUpload),
        crop: state.crop ? { ...state.crop } : null, undoCrops: state.previous.length,
    }),
    // Для проверок интерфейса: кадрировать без мыши.
    crop: async (rect, ratio = '') => {
        startCrop();
        setCropRatio(ratio);
        if (rect) { state.crop = clampRect(rect); showCrop(); }
        await applyCrop();
    },
    destroy: () => {
        listeners.forEach(off => off());
        resizeObserver.disconnect();
        mutationObserver.disconnect();
        win.clearTimeout(state.syncTimer);
    },
};
element.__qsPainter = api;
// Имя — elem_id блока. Искать от родителя: у самого узла компонента Gradio
// свой служебный id вида html-…, и ближайшим с id оказался бы он.
const block = element.parentElement && element.parentElement.closest('[id]');
win.__qsPainters = win.__qsPainters || {};
win.__qsPainters[block ? block.id : 'painter'] = api;
