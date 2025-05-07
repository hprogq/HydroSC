#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hydroSC.py  –  HydroOJ Similarity Checker
"""

import os, sys, re, json, zipfile, shutil, traceback, base64
from collections import defaultdict
from difflib import SequenceMatcher

# ─── 第三方 ───────────────────────────────────
try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    import requests, bs4, openpyxl
except ImportError:
    print("缺少依赖，请执行:\n  pip install PyQt5 requests beautifulsoup4 openpyxl")
    sys.exit(1)

APP_TITLE   = "HydroOJ Similarity Checker"
CACHE_DIR   = "cache"
ANS_DIR     = "answers"
CRED_FILE   = os.path.join(CACHE_DIR, "cred.dat")
DEF_THRESH  = 1.00   # 默认阈 100%

# ────────────────────── 工具函数 ──────────────────────
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def log_exc(): traceback.print_exc()

def strip_comments(code: str, lang: str) -> str:
    """去注释"""
    patterns = {
        'c'   : r'//.*?$|/\*.*?\*/',
        'cpp' : r'//.*?$|/\*.*?\*/',
        'java': r'//.*?$|/\*.*?\*/',
        'py'  : r'#.*?$|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'',
    }
    patt = patterns.get(lang, '')
    return re.sub(patt, '', code, flags=re.S | re.M) if patt else code

def normalize(code: str, lang: str) -> str:
    """统一大小写、空白、去注释"""
    code = strip_comments(code, lang)
    code = re.sub(r'\s+', ' ', code)
    return code.lower().strip()

# ────────────────────── 相似度阈值对话框 ────────────────
class ThresholdDialog(QtWidgets.QDialog):
    def __init__(self, default: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择相似度阈值")
        self.resize(300, 120)

        v = QtWidgets.QVBoxLayout(self)

        # 滑块
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(50, 100)  # 50% ~ 100%
        self.slider.setValue(int(default * 100))

        # 文本框
        self.edit = QtWidgets.QLineEdit(f"{default:.2f}")
        self.edit.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 2, self))

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("阈值 (0‑1):"))
        h.addWidget(self.edit, 1)

        # 按钮
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)

        v.addWidget(self.slider)
        v.addLayout(h)
        v.addWidget(bb)

        # 同步事件
        self.slider.valueChanged.connect(self._from_slider)
        self.edit.textEdited.connect(self._from_edit)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)

    def _from_slider(self, val: int):
        self.edit.setText(f"{val/100:.2f}")

    def _from_edit(self, _txt: str):
        try:
            f = float(self.edit.text())
            self.slider.setValue(int(f * 100))
        except ValueError:
            pass

    def value(self) -> float:
        return float(self.edit.text() or "0")

# ────────────────────── 查重核心 ────────────────
class PlagiarismChecker:
    LANG_EXT = {
        '.c': 'c', '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp',
        '.java': 'java', '.py': 'py', '.py3': 'py'
    }

    def __init__(self, ans_dir: str, hwid: str, threshold: float):
        self.dir = ans_dir
        self.hwid = hwid
        self.thr = threshold
        self.subs = []

    def load(self):
        """遍历 answers/ 目录收集提交"""
        for root, _, files in os.walk(self.dir):
            for fn in files:
                m = re.match(r'^U(\d+)_P(\d+)_R([0-9a-fA-F]+)', fn, re.I)
                if not m:
                    continue
                uid, pid, rec = m.groups()

                # 语言判定
                ext = os.path.splitext(fn)[1].lower()
                lang = 'cpp' if ext.startswith('.cc') or '.cpp' in fn else self.LANG_EXT.get(ext)
                if not lang:
                    continue

                path = os.path.join(root, fn)
                try:
                    code = open(path, 'r', encoding='utf-8', errors='ignore').read()
                except Exception:
                    continue

                self.subs.append(
                    dict(user=uid, pid=pid, sub=rec, lang=lang,
                         code=code, norm=normalize(code, lang), path=path)
                )

        if not self.subs:
            raise RuntimeError("无有效代码")

    def sim(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def cluster(self):
        """按 (pid,lang) 先分桶，再聚类"""
        buckets = defaultdict(list)
        for i, s in enumerate(self.subs):
            buckets[(s['pid'], s['lang'])].append(i)

        result = defaultdict(lambda: defaultdict(list))

        for (pid, lang), idxs in buckets.items():
            n = len(idxs)
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            simtbl = {}
            for i in range(n):
                for j in range(i + 1, n):
                    ai, aj = self.subs[idxs[i]], self.subs[idxs[j]]
                    # 长度过滤
                    if min(len(ai['norm']), len(aj['norm'])) / max(len(ai['norm']), len(aj['norm'])) < 0.6:
                        continue
                    s = self.sim(ai['norm'], aj['norm'])
                    if s >= self.thr:
                        union(i, j)
                        simtbl[(i, j)] = s

            groups = defaultdict(list)
            for i in range(n):
                groups[find(i)].append(i)

            for gid, g in enumerate(groups.values(), 1):
                if len(g) < 2:
                    continue
                members = []
                for i in g:
                    best = max([simtbl.get(tuple(sorted((i, j))), 0) for j in g if j != i] + [1])
                    s = self.subs[idxs[i]]
                    members.append(dict(
                        user=s['user'], sub=s['sub'],
                        similarity=round(best, 3),
                        code=s['code'], path=s['path']
                    ))
                result[pid][lang].append(dict(groupId=gid, members=members))
        return result

    def run(self):
        self.load()
        data = self.cluster()
        ensure_dir(CACHE_DIR)
        fp = os.path.join(CACHE_DIR, f"result-{self.hwid}.json")
        json.dump(data, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        return fp

# ────────────────────── 后台线程 ────────────────
class Worker(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)

    def __init__(self, fn, *a, **kw):
        super().__init__()
        self.fn = fn
        self.a = a
        self.kw = kw

    def run(self):
        try:
            res = self.fn(*self.a, **self.kw)
            self.finished.emit(res)
        except Exception as e:
            log_exc()
            self.finished.emit(e)

# ────────────────────── 主窗口 ───────────────────
class MainWindow(QtWidgets.QMainWindow):
    # ——— 初始化 ———
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1100, 700)
        ensure_dir(CACHE_DIR)

        # 状态
        self.session = requests.Session()
        self.user_map = {}
        self.host = self.domain_id = self.homework_id = ""
        self.threshold = DEF_THRESH

        # 页面栈
        self._hist = []   # 后退栈
        self._fwd  = []   # 前进栈

        # 中央堆栈
        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self._init_pages()
        self._init_nav()
        self.stack.setCurrentWidget(self.pg_login)
        self.load_cred()
        self.update_nav()

    # ——— 顶部导航栏 ———
    def _init_nav(self):
        tb = self.addToolBar("nav")
        tb.setMovable(False)

        self.act_home = tb.addAction("返回首页", lambda: self.goto(self.pg_login, clear_forward=True))
        self.act_prev = tb.addAction("上一步", self.go_prev)
        self.act_next = tb.addAction("下一步", self.go_next)

    def record(self, widget):
        """记录历史，清空前进栈"""
        if self._hist and self._hist[-1] is widget:
            return
        self._hist.append(widget)
        self._fwd.clear()

    def goto(self, widget, clear_forward=False):
        """主导航切换"""
        current = self.stack.currentWidget()
        if current is widget:
            return
        self.record(current)
        if clear_forward:
            self._fwd.clear()
        self.stack.setCurrentWidget(widget)
        self.update_nav()

    def go_prev(self):
        if not self._hist:
            return
        self._fwd.append(self.stack.currentWidget())
        self.stack.setCurrentWidget(self._hist.pop())
        self.update_nav()

    def go_next(self):
        if not self._fwd:
            return
        self._hist.append(self.stack.currentWidget())
        self.stack.setCurrentWidget(self._fwd.pop())
        self.update_nav()

    def update_nav(self):
        """根据当前页刷新按钮状态"""
        on_home = (self.stack.currentWidget() is self.pg_login)
        self.act_prev.setVisible(not on_home and bool(self._hist))
        self.act_next.setEnabled(bool(self._fwd))

    # ——— 页面初始化 ———
    def _init_pages(self):
        self.pg_login  = self._page_login();   self.stack.addWidget(self.pg_login)
        self.pg_domain = self._page_domain();  self.stack.addWidget(self.pg_domain)
        self.pg_hw     = self._page_hw();      self.stack.addWidget(self.pg_hw)
        self.pg_prog   = self._page_prog();    self.stack.addWidget(self.pg_prog)
        self.pg_res    = self._page_res();     self.stack.addWidget(self.pg_res)

    # ——— URL 前缀 ———
    def _prefix(self) -> str:
        return '' if self.domain_id == 'system' else f"d/{self.domain_id}/"

    # ——— 登录页 ———
    def _page_login(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)

        self.in_host = QtWidgets.QLineEdit()
        self.in_user = QtWidgets.QLineEdit()
        self.in_pass = QtWidgets.QLineEdit(); self.in_pass.setEchoMode(2)

        btn_log = QtWidgets.QPushButton("登录")
        btn_off = QtWidgets.QPushButton("离线查重（选择导出的代码包）")

        f.addRow("服务器地址", self.in_host)
        f.addRow("用户名", self.in_user)
        f.addRow("密码", self.in_pass)

        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(btn_log)
        hl.addWidget(btn_off)
        f.addRow(hl)

        btn_log.clicked.connect(self.do_login)
        btn_off.clicked.connect(self.choose_offline)

        return w

    # 读取保存的登录信息
    def load_cred(self):
        if os.path.isfile(CRED_FILE):
            try:
                j = json.loads(base64.b64decode(open(CRED_FILE, 'rb').read()))
                self.in_host.setText(j.get("host", ""))
                self.in_user.setText(j.get("user", ""))
                self.in_pass.setText(j.get("pass_", ""))
            except Exception:
                pass

    def save_cred(self):
        data = dict(host=self.host, user=self.in_user.text(), pass_=self.in_pass.text())
        enc = base64.b64encode(json.dumps(data).encode())
        open(CRED_FILE, 'wb').write(enc)

    # 登录动作
    def do_login(self):
        self.host = self.in_host.text().strip().rstrip('/')
        u = self.in_user.text().strip()
        p = self.in_pass.text()

        if not all((self.host, u, p)):
            QtWidgets.QMessageBox.warning(self, "提示", "请填完整信息")
            return

        try:
            r = self.session.post(
                f"{self.host}/login",
                data=dict(uname=u, password=p, tfa="", authnChallenge=""),
                allow_redirects=False,
                timeout=10
            )
            if not self.session.cookies.get('sid'):
                raise RuntimeError("登录失败，检查用户名/密码")

            QtWidgets.QMessageBox.information(self, "成功", "登录成功！")
            self.save_cred()
            self.load_domains()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", str(e))

    # ——— 域选择页 ———
    def _page_domain(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        self.lst_dom = QtWidgets.QListWidget()
        btn_next = QtWidgets.QPushButton("下一步")

        v.addWidget(QtWidgets.QLabel("选择域"))
        v.addWidget(self.lst_dom, 1)
        v.addWidget(btn_next)

        btn_next.clicked.connect(self.sel_domain)
        return w

    def load_domains(self):
        try:
            r = self.session.get(f"{self.host}/home/domain", timeout=10)
            soup = bs4.BeautifulSoup(r.text, "html.parser")

            self.lst_dom.clear()

            seen = {}
            for td in soup.select('td.col--name'):
                attr = td.get('data-star-action')
                m = re.search(r'id=([\w\-]+)', attr or '')
                if not m:
                    continue
                did = m.group(1)

                name = td.get_text(strip=True)
                it = QtWidgets.QListWidgetItem(name)
                it.setData(32, did)
                self.lst_dom.addItem(it)

            self.goto(self.pg_domain)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", str(e))

    def sel_domain(self):
        it = self.lst_dom.currentItem()
        if it:
            self.domain_id = it.data(32)
            self.load_homeworks()

    # ——— 作业选择页 ———
    def _page_hw(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        self.lst_hw = QtWidgets.QListWidget()
        btn_dl = QtWidgets.QPushButton("下载并查重")

        v.addWidget(QtWidgets.QLabel("选择作业"))
        v.addWidget(self.lst_hw, 1)
        v.addWidget(btn_dl)

        btn_dl.clicked.connect(self.start_check_online)
        return w

    def load_homeworks(self):
        try:
            r = self.session.get(f"{self.host}/{self._prefix()}homework", timeout=10)
            soup = bs4.BeautifulSoup(r.text, "html.parser")

            self.lst_hw.clear()
            pat = rf"^/{self._prefix()}homework/([\w\d]+)$"

            for a in soup.find_all("a", href=True):
                m = re.match(pat, a['href'])
                if m:
                    hid = m.group(1)
                    if hid == "create":
                        continue;
                    name = a.get_text(strip=True)
                    it = QtWidgets.QListWidgetItem(f"{name} ({hid})")
                    it.setData(32, hid)
                    self.lst_hw.addItem(it)

            if not self.lst_hw.count():
                raise RuntimeError("当前域下无作业")

            self.goto(self.pg_hw)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", str(e))

    # ——— 进度页 ———
    def _page_prog(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        self.lb_prog = QtWidgets.QLabel("处理中…")
        self.pb = QtWidgets.QProgressBar()
        self.pb.setRange(0, 0)

        v.addWidget(self.lb_prog, 0, QtCore.Qt.AlignCenter)
        v.addWidget(self.pb)
        return w

    def show_prog(self, msg: str):
        self.lb_prog.setText(msg)
        self.goto(self.pg_prog)

    # ——— 结果页 ———
    def _page_res(self):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)

        # 左栏：导入按钮 + Tree
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        self.btn_excel = QtWidgets.QPushButton("导入 Excel 名单")
        self.btn_excel.clicked.connect(self.import_excel)
        self.tree = QtWidgets.QTreeWidget(); self.tree.setHeaderLabels(["雷同结果（已保存至 cache 文件夹）"])
        self.tree.itemClicked.connect(self.tree_click)
        lv.addWidget(self.btn_excel)
        lv.addWidget(self.tree, 1)
        h.addWidget(left, 2)

        # 右栏：Splitter(成员列表 + 源码)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.lst_mem = QtWidgets.QListWidget(); self.lst_mem.itemClicked.connect(self.mem_click)
        self.code_view = QtWidgets.QPlainTextEdit(); self.code_view.setReadOnly(True)
        self.splitter.addWidget(self.lst_mem)
        self.splitter.addWidget(self.code_view)
        self.splitter.setStretchFactor(1, 5)
        h.addWidget(self.splitter, 5)
        return w

    # 导入 Excel 名单
    def import_excel(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择名单", "", "Excel (*.xlsx *.xls)")
        if not fn:
            return
        try:
            wb = openpyxl.load_workbook(fn)
            ws = wb.active
            self.user_map.update({
                str(r[0].value): f"{r[1].value}({r[0].value})"
                for r in ws.iter_rows(min_row=2) if r[0].value
            })
            self.load_result()
            QtWidgets.QMessageBox.information(self, "OK", "名单已导入并刷新")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", str(e))

    # ——— 阈值对话框 ———
    def ask_threshold(self) -> bool:
        dlg = ThresholdDialog(self.threshold, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.threshold = dlg.value()
            return True
        return False

    # ——— 在线查重流程 ———
    def start_check_online(self):
        item = self.lst_hw.currentItem()
        if not item:
            return
        if not self.ask_threshold():
            return

        self.homework_id = item.data(32)
        self.show_prog("下载作业代码…")

        self.worker = Worker(self._job_online)
        self.worker.finished.connect(self.job_done)
        self.worker.start()

    def _job_online(self):
        ensure_dir(CACHE_DIR)
        zip_path = os.path.join(CACHE_DIR, f"{self.homework_id}.zip")

        # 下载 ZIP（若本地不存在）
        if not os.path.isfile(zip_path):
            url = f"{self.host}/{self._prefix()}homework/{self.homework_id}/code"
            with self.session.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                open(zip_path, 'wb').write(r.content)

        # 解压
        if os.path.isdir(ANS_DIR):
            shutil.rmtree(ANS_DIR)
        zipfile.ZipFile(zip_path).extractall(ANS_DIR)

        # 查重
        result_json = PlagiarismChecker(ANS_DIR, self.homework_id,
                                        self.threshold).run()

        # 获取用户名
        data = json.load(open(result_json, encoding='utf-8'))
        ids = {
            m['user'] for pid in data for lang in data[pid]
            for g in data[pid][lang] for m in g['members']
        }
        if ids:
            api = f"{self.host}/{self._prefix()}api/users"
            payload = json.dumps({
                "args": {"auto": list(ids)},
                "projection": ["_id", "uname", "displayName"]
            })
            r = self.session.post(api, data=payload,
                                  headers={'Content-Type': 'application/json'},
                                  timeout=20)
            self.user_map = {
                str(u['_id']): (u.get('displayName') or u['uname']) +
                               f"({u['_id']})"
                for u in r.json()
            }

        return result_json

    # ——— 离线查重流程 ———
    def choose_offline(self):
        zp, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择代码 ZIP", "", "Zip (*.zip)")
        if not zp:
            return
        if not self.ask_threshold():
            return

        self.homework_id = os.path.splitext(os.path.basename(zp))[0]
        self.show_prog("离线查重…")

        self.worker = Worker(self._job_offline, zp)
        self.worker.finished.connect(self.job_done)
        self.worker.start()

    def _job_offline(self, zip_path: str):
        if os.path.isdir(ANS_DIR):
            shutil.rmtree(ANS_DIR)
        zipfile.ZipFile(zip_path).extractall(ANS_DIR)
        return PlagiarismChecker(ANS_DIR, self.homework_id,
                                 self.threshold).run()

    # ——— 任务结束统一回调 ———
    def job_done(self, res):
        if isinstance(res, Exception):
            QtWidgets.QMessageBox.critical(self, "错误", str(res))
            self.goto(self.pg_login)
            return
        self.json_path = res
        self.load_result()

    # ——— 结果树渲染 ———
    def load_result(self):
        self.tree.clear()
        self.lst_mem.clear()
        self.code_view.clear()

        data = json.load(open(self.json_path, encoding='utf-8'))
        for pid in sorted(data, key=lambda x: int(x)):
            tpid = QtWidgets.QTreeWidgetItem(self.tree, [f"题目 {pid}"])
            for lang in data[pid]:
                tlang = QtWidgets.QTreeWidgetItem(tpid, [lang])
                for g in data[pid][lang]:
                    gn = QtWidgets.QTreeWidgetItem(
                        tlang, [f"组 {g['groupId']} ({len(g['members'])})"])
                    gn.setData(0, 32, g['members'])
                tlang.setExpanded(True)
            tpid.setExpanded(True)

        self.goto(self.pg_res)

    # ——— Tree / List 交互 ———
    def tree_click(self, it, _col):
        mems = it.data(0, 32)
        if mems:
            self.show_group(mems)

    def show_group(self, mems):
        self.lst_mem.clear()
        self.code_view.clear()
        for m in mems:
            name = self.user_map.get(m['user'], m['user'])
            item = QtWidgets.QListWidgetItem(name)
            item.setData(32, m)
            self.lst_mem.addItem(item)

        if self.lst_mem.count():
            self.lst_mem.setCurrentRow(0)
            self.mem_click(self.lst_mem.item(0))

    def mem_click(self, item):
        mem = item.data(32)
        if mem:
            self.code_view.setPlainText(mem['code'])
            self.code_view.moveCursor(QtGui.QTextCursor.Start)

# ────────────────────── 入口 ───────────────────
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()