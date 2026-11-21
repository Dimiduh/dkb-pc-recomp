"""Unit tests for parallel slice collection: compact snapshots, inline binds, and DLSS pass latch.

Extracts ParPassSnap, ParSnap, par_pass_take/load, par_snap_take/load and par_bind_op verbatim from
runtime/vk_exec.c, plus vk_dlss_scene_draw from runtime/vk_dlss.c, and verifies them at -O0 and -O2.
"""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def extract_typedef(source, name):
    end = source.index(f'}} {name};') + len(f'}} {name};')
    start = source.rindex('typedef struct {', 0, end)
    return source[start:end]


def extract_function(source, signature):
    start = source.index(signature)
    brace = source.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def extract_macro(source, name):
    m = re.search(rf'^#define {re.escape(name)}\b.*?(?:\\\n.*?)*$', source, re.M)
    if not m:
        raise ValueError(f'macro {name} not found')
    return m.group(0)


HARNESS = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vk_exec.h"
#include "vk_spec.h"

/* ---- stubs for par_bind_op fallback ---- */
static int g_fallback_calls;
static uint32_t g_fallback_last_op;

void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload) {
    (void)ex;
    (void)payload;
    g_fallback_calls++;
    g_fallback_last_op = op;
}

/* ---- extracted vk_exec.c types and functions ---- */
/* VK_EXEC_EXTRACT */

/* ---- stubs and state for vk_dlss_scene_draw ---- */
#define CAM_BLOCK 0x100u
typedef struct {
    int have;
    uint32_t serial;
    float cam[64];
    VkTexImage z;
    VkFormat zfmt;
    uint32_t zlevel, zlayer;
} Scene;

static int state = 1;
static struct { uint32_t w, h; uint64_t frame; } J;
static Scene S, SH[VK_SPEC_HELPERS];
static int g_fsr_wanted = 0;
static int g_dlss_wanted = 1;
static int g_tex_locks = 0;
static const void *g_usnap_ptr = NULL;
static uint64_t g_usnap_addr = 0;
static int g_usnap_calls = 0;

int vk_fsr3_wanted(void) { return g_fsr_wanted; }
int vk_fsr4_wanted(void) { return 0; }
int vk_dlss_wanted(void) { return g_dlss_wanted; }
static uint64_t frames_now(void) { return 100; }
const char *cfg_value(const char *k) { (void)k; return "dlss"; }
void rt_log(const char *fmt, ...) { (void)fmt; }
const void *vk_usnap_bytes(uint64_t addr, size_t n) {
    (void)n;
    g_usnap_calls++;
    return addr == g_usnap_addr ? g_usnap_ptr : NULL;
}
void vk_texture_lock(void) { g_tex_locks++; }
void vk_texture_unlock(void) { g_tex_locks--; }
_Thread_local int vk_spec_here = 0;

/* ---- extracted vk_dlss.c function ---- */
/* VK_DLSS_EXTRACT */

static void test_pass_snap(void) {
    VkExec ex, w;
    memset(&ex, 0, sizeof ex);
    memset(&w, 0x5a, sizeof w);

    ex.fence = 42;
    ex.submitted_ticks = 9999;
    ex.in_pass = 1;
    ex.rt_resolved = 1;
    ex.render_disabled = 0;
    ex.has_render_cond = 1;
    ex.pass_w = 1920;
    ex.pass_h = 1080;
    ex.pass_layers = 2;
    ex.pass_noatt = 0;
    ex.pass_ncolor = 2;
    ex.pass_samples = 1;
    ex.pass_color[0] = VK_FORMAT_R8G8B8A8_UNORM;
    ex.pass_color[1] = VK_FORMAT_R16G16B16A16_SFLOAT;
    ex.pass_depth = VK_FORMAT_D32_SFLOAT;
    ex.pass_stencil = VK_FORMAT_UNDEFINED;
    ex.color[0].image = (VkImage)(uintptr_t)0x1111;
    ex.color[0].width = 1920;
    ex.color[1].image = (VkImage)(uintptr_t)0x2222;
    ex.depth.image = (VkImage)(uintptr_t)0x3333;
    ex.depth.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    ex.fbl_img[0] = (VkImage)(uintptr_t)0x1111;
    ex.fbl_aspect[0] = VK_IMAGE_ASPECT_COLOR_BIT;
    ex.nfbl = 1;
    ex.fbl_mask = 1;
    ex.pass_serial = 77;

    ParPassSnap ps;
    memset(&ps, 0xcd, sizeof ps);
    par_pass_take(&ps, &ex);
    par_pass_load(&w, &ps);

    assert(w.fence == 42);
    assert(w.submitted_ticks == 9999);
    assert(w.in_pass == 1 && w.rt_resolved == 1);
    assert(w.render_disabled == 0 && w.has_render_cond == 1);
    assert(w.pass_w == 1920 && w.pass_h == 1080 && w.pass_layers == 2);
    assert(w.pass_ncolor == 2 && w.pass_samples == 1);
    assert(w.pass_color[0] == VK_FORMAT_R8G8B8A8_UNORM);
    assert(w.pass_color[1] == VK_FORMAT_R16G16B16A16_SFLOAT);
    assert(w.pass_depth == VK_FORMAT_D32_SFLOAT);
    assert(w.color[0].image == (VkImage)(uintptr_t)0x1111);
    assert(w.color[1].image == (VkImage)(uintptr_t)0x2222);
    assert(w.depth.image == (VkImage)(uintptr_t)0x3333);
    assert(w.fbl_img[0] == (VkImage)(uintptr_t)0x1111);
    assert(w.nfbl == 1 && w.fbl_mask == 1 && w.pass_serial == 77);
}

static void test_slice_snap_roundtrip_and_shrink(void) {
    VkExec ex, w;
    ParSnap s1, s2, s3;
    memset(&ex, 0, sizeof ex);
    memset(&w, 0, sizeof w);

    /* Slice 1: populated slots across all 3 graphics stages, viewports, vbufs, push constants. */
    ex.tex_pool.addr = 0x10000;
    ex.samp_pool.addr = 0x20000;
    ex.has_vp = 1;
    ex.has_scissor = 1;
    ex.scissor.x = 10; ex.scissor.y = 20; ex.scissor.w = 640; ex.scissor.h = 360;
    ex.depth_range[0] = 0.0f; ex.depth_range[1] = 1.0f;
    ex.point_size = 4.0f;
    ex.blend_color[0] = 0.25f; ex.blend_color[3] = 0.75f;
    ex.st_vmask[0] = 0xaa; ex.st_wmask[1] = 0x55; ex.st_ref[0] = 7;
    ex.poly_offset.factor = 1.5f;
    ex.prim_restart.enable = 1; ex.prim_restart.index = 0xffffu;
    ex.pipe.depth_clamp = 1;
    ex.prog = (const prog_t *)(uintptr_t)0xfeedface;
    ex.prog_stages = 0x3;
    ex.push_set[0][0] = 0x5ull;
    ex.push[0][0][0] = 0x12345678u;
    ex.push[0][0][2] = 0x9abcdef0u;

    ex.vp_hi = 4;
    for (int i = 0; i < 4; i++) ex.vp[i][2] = (float)(100 + i);
    ex.vbuf_hi = 5;
    for (int i = 0; i < 5; i++) ex.vbuf[i].address = 0x5000 + (uint64_t)i * 64, ex.vbuf[i].size = 64;
    for (int st = 0; st < 3; st++) {
        ex.ubo_hi[st] = (uint8_t)(6 - st);
        for (int i = 0; i < ex.ubo_hi[st]; i++) ex.ubo[st][i].addr = 0x1000 + (uint64_t)st * 100 + (uint64_t)i;
        ex.ssbo_hi[st] = (uint8_t)(4 - st);
        for (int i = 0; i < ex.ssbo_hi[st]; i++) ex.ssbo[st][i].addr = 0x2000 + (uint64_t)st * 100 + (uint64_t)i;
        ex.tex_hi[st] = (uint8_t)(10 - st * 2);
        for (int i = 0; i < ex.tex_hi[st]; i++) ex.tex[st][i] = 0x3000 + (uint64_t)st * 100 + (uint64_t)i;
        ex.samp_hi[st] = (uint8_t)(8 - st * 2);
        for (int i = 0; i < ex.samp_hi[st]; i++) ex.samp[st][i] = 0x4000 + (uint64_t)st * 100 + (uint64_t)i;
        ex.img_hi[st] = (uint8_t)(3 - st);
        for (int i = 0; i < ex.img_hi[st]; i++) ex.img[st][i] = 0x5000 + (uint64_t)st * 100 + (uint64_t)i;
    }

    memset(&s1, 0xdd, sizeof s1);
    par_snap_take(&s1, &ex);
    par_snap_load(&w, &s1);

    assert(w.tex_pool.addr == 0x10000 && w.samp_pool.addr == 0x20000);
    assert(w.has_vp == 1 && w.has_scissor == 1 && w.scissor.w == 640);
    assert(w.point_size == 4.0f && w.blend_color[3] == 0.75f);
    assert(w.st_vmask[0] == 0xaa && w.st_wmask[1] == 0x55 && w.st_ref[0] == 7);
    assert(w.pipe.depth_clamp == 1 && w.prog == ex.prog && w.prog_stages == 0x3);
    assert(w.push_set[0][0] == 0x5ull && w.push[0][0][0] == 0x12345678u && w.push[0][0][2] == 0x9abcdef0u);
    assert(w.vp_hi == 4 && w.vp[3][2] == 103.0f);
    assert(w.vbuf_hi == 5 && w.vbuf[4].address == 0x5000 + 4 * 64);
    for (int st = 0; st < 3; st++) {
        assert(w.ubo_hi[st] == ex.ubo_hi[st] && w.ssbo_hi[st] == ex.ssbo_hi[st]);
        assert(w.tex_hi[st] == ex.tex_hi[st] && w.samp_hi[st] == ex.samp_hi[st] && w.img_hi[st] == ex.img_hi[st]);
        for (int i = 0; i < w.ubo_hi[st]; i++) assert(w.ubo[st][i].addr == ex.ubo[st][i].addr);
        for (int i = 0; i < w.tex_hi[st]; i++) assert(w.tex[st][i] == ex.tex[st][i]);
    }

    /* Simulate slice 1 replay raising worker high-water marks even further. */
    w.vp[9][2] = 999.0f; w.vp_hi = 10;
    w.vbuf[11].address = 0xdeadbeef; w.vbuf_hi = 12;
    w.ubo[0][13].addr = 0xbeef; w.ubo_hi[0] = 14;
    w.ssbo[1][9].addr = 0xcafe; w.ssbo_hi[1] = 10;
    w.tex[1][40] = 0x7777; w.tex_hi[1] = 41;
    w.samp[2][25] = 0x8888; w.samp_hi[2] = 26;
    w.img[0][7] = 0x9999; w.img_hi[0] = 8;

    /* Slice 2: much smaller high-water marks (e.g. after state_reset), push_set0 == 0. */
    memset(&ex, 0, sizeof ex);
    ex.vp_hi = 1; ex.vp[0][2] = 50.0f;
    ex.vbuf_hi = 2; ex.vbuf[1].address = 0x7000;
    ex.ubo_hi[0] = 2; ex.ubo[0][1].addr = 0x8001;
    ex.ssbo_hi[1] = 1; ex.ssbo[1][0].addr = 0x8002;
    ex.tex_hi[1] = 3; ex.tex[1][2] = 0x8003;
    ex.samp_hi[2] = 1; ex.samp[2][0] = 0x8004;
    ex.img_hi[0] = 1; ex.img[0][0] = 0x8005;

    par_snap_take(&s2, &ex);
    par_snap_load(&w, &s2);

    assert(w.push_set[0][0] == 0);
    assert(w.vp_hi == 1 && w.vp[0][2] == 50.0f && w.vp[1][2] == 0.0f && w.vp[9][2] == 0.0f);
    assert(w.vbuf_hi == 2 && w.vbuf[1].address == 0x7000 && w.vbuf[2].address == 0 && w.vbuf[11].address == 0);
    assert(w.ubo_hi[0] == 2 && w.ubo[0][1].addr == 0x8001 && w.ubo[0][2].addr == 0 && w.ubo[0][13].addr == 0);
    assert(w.ubo_hi[1] == 0 && w.ubo[1][0].addr == 0);
    assert(w.ssbo_hi[1] == 1 && w.ssbo[1][0].addr == 0x8002 && w.ssbo[1][9].addr == 0);
    assert(w.tex_hi[1] == 3 && w.tex[1][2] == 0x8003 && w.tex[1][3] == 0 && w.tex[1][40] == 0);
    assert(w.samp_hi[2] == 1 && w.samp[2][0] == 0x8004 && w.samp[2][25] == 0);
    assert(w.img_hi[0] == 1 && w.img[0][0] == 0x8005 && w.img[0][7] == 0);

    /* Slice 3: completely empty high-water marks clear the remaining active prefix. */
    memset(&ex, 0, sizeof ex);
    par_snap_take(&s3, &ex);
    par_snap_load(&w, &s3);
    assert(w.vp_hi == 0 && w.vp[0][2] == 0.0f);
    assert(w.vbuf_hi == 0 && w.vbuf[1].address == 0);
    assert(w.ubo_hi[0] == 0 && w.ubo[0][1].addr == 0);
    assert(w.tex_hi[1] == 0 && w.tex[1][2] == 0);
    assert(w.img_hi[0] == 0 && w.img[0][0] == 0);
}

static void test_par_bind_op(void) {
    VkExec ex;
    memset(&ex, 0, sizeof ex);
    g_fallback_calls = 0;

    p_bind_program bp = {.prog = (const prog_t *)(uintptr_t)0x1234, .stages = 0x7};
    par_bind_op(&ex, OP_BIND_PROGRAM, &bp);
    assert(ex.prog == bp.prog && ex.prog_stages == 0x7);

    p_bind_state bs = {0};
    blend_st bl = {.target = 2, .src_rgb = 5};
    bs.kind = 0; memcpy(bs.bytes, &bl, sizeof bl);
    par_bind_op(&ex, OP_BIND_STATE, &bs);
    assert(ex.pipe.blend[2].src_rgb == 5);
    float ps = 3.5f;
    bs.kind = 6; memcpy(bs.bytes, &ps, sizeof ps);
    par_bind_op(&ex, OP_BIND_STATE, &bs);
    assert(ex.point_size == 3.5f);

    p_bind_vattr va = {.first = 1, .n = 2, .st = {0x11, 0x22}};
    ex.pipe.vattr[5] = 0x99;
    par_bind_op(&ex, OP_BIND_VATTR, &va);
    assert(ex.pipe.vattr[1] == 0x11 && ex.pipe.vattr[2] == 0x22 && ex.pipe.vattr[5] == 0);

    p_bind_vstream vs = {.first = 0, .n = 1};
    vs.st[0].stride = 32;
    ex.pipe.vstream[3].stride = 64;
    par_bind_op(&ex, OP_BIND_VSTREAM, &vs);
    assert(ex.pipe.vstream[0].stride == 32 && ex.pipe.vstream[3].stride == 0);

    p_bind_vbuf vb = {.first = 2, .n = 2};
    vb.r[0].address = 0xa000; vb.r[1].address = 0xb000;
    par_bind_op(&ex, OP_BIND_VBUF, &vb);
    assert(ex.vbuf_hi == 4 && ex.vbuf[2].address == 0xa000 && ex.vbuf[3].address == 0xb000);

    p_bind_buf bu = {.stage = 1, .index = 4, .addr = 0xc000, .size = 256};
    par_bind_op(&ex, OP_BIND_UBO, &bu);
    assert(ex.ubo_hi[1] == 5 && ex.ubo[1][4].addr == 0xc000);

    /* Stage 5 (compute) updates slot without touching graphics high-water marks [0..2]. */
    p_bind_buf bu_cs = {.stage = 5, .index = 10, .addr = 0xc500, .size = 64};
    par_bind_op(&ex, OP_BIND_UBO, &bu_cs);
    assert(ex.ubo[5][10].addr == 0xc500);
    assert(ex.ubo_hi[0] == 0 && ex.ubo_hi[1] == 5 && ex.ubo_hi[2] == 0);

    p_bind_buf bsbo = {.stage = 2, .index = 3, .addr = 0xd000, .size = 128};
    par_bind_op(&ex, OP_BIND_SSBO, &bsbo);
    assert(ex.ssbo_hi[2] == 4 && ex.ssbo[2][3].addr == 0xd000);

    p_bind_handles bh = {.stage = 0, .first = 4, .n = 3, .h = {11, 22, 33}};
    par_bind_op(&ex, OP_BIND_TEX, &bh);
    assert(ex.tex_hi[0] == 7 && ex.tex[0][4] == 11 && ex.tex[0][6] == 33);

    bh.stage = 1; bh.first = 2; bh.n = 2;
    par_bind_op(&ex, OP_BIND_SAMP, &bh);
    assert(ex.samp_hi[1] == 4 && ex.samp[1][2] == 11 && ex.samp[1][3] == 22);

    bh.stage = 2; bh.first = 1; bh.n = 2;
    par_bind_op(&ex, OP_BIND_IMG, &bh);
    assert(ex.img_hi[2] == 3 && ex.img[2][1] == 11 && ex.img[2][2] == 22);

    p_push_const pc = {.compute = 0, .table = 0, .offset = 8, .bytes = 8, .w = {0xaaaau, 0xbbbbu}};
    par_bind_op(&ex, OP_PUSH_CONST, &pc);
    assert(ex.push_set[0][0] == (3ull << 2) && ex.push[0][0][2] == 0xaaaau && ex.push[0][0][3] == 0xbbbbu);

    assert(g_fallback_calls == 0);

    /* Out-of-range binds must fall back to vk_exec_op_draw so warnings fire. */
    p_bind_buf bad_ssbo = {.stage = 0, .index = 16, .addr = 0x100, .size = 16};
    par_bind_op(&ex, OP_BIND_SSBO, &bad_ssbo);
    assert(g_fallback_calls == 1 && g_fallback_last_op == OP_BIND_SSBO);

    p_bind_handles bad_tex = {.stage = 0, .first = 120, .n = 16};
    par_bind_op(&ex, OP_BIND_TEX, &bad_tex);
    assert(g_fallback_calls == 2 && g_fallback_last_op == OP_BIND_TEX);

    p_bind_handles bad_img = {.stage = 0, .first = 30, .n = 4};
    par_bind_op(&ex, OP_BIND_IMG, &bad_img);
    assert(g_fallback_calls == 3 && g_fallback_last_op == OP_BIND_IMG);

    p_push_const bad_pc = {.compute = 0, .table = 1, .offset = 0, .bytes = 4};
    par_bind_op(&ex, OP_PUSH_CONST, &bad_pc);
    assert(g_fallback_calls == 4 && g_fallback_last_op == OP_PUSH_CONST);
}

static void test_dlss_latch(void) {
    VkExec ex;
    VkTexImage zimg;
    float cam_data[64];
    for (int i = 0; i < 64; i++) cam_data[i] = (float)(i + 1);

    memset(&ex, 0, sizeof ex);
    memset(&zimg, 0, sizeof zimg);
    memset(&S, 0, sizeof S);
    state = 1; g_fsr_wanted = 0; g_dlss_wanted = 1;
    J.w = 1920; J.h = 1080; J.frame = 99;

    zimg.image = (VkImage)(uintptr_t)0x9000;
    ex.color[0].image = (VkImage)(uintptr_t)0x8000;
    ex.depth.image = zimg.image;
    ex.depth.img = &zimg;
    ex.depth.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    ex.depth.width = 1920; ex.depth.height = 1080; ex.depth.samples = 1;
    ex.pass_serial = 11;

    /* Ineligible target (shadow map size mismatch): returns 1 immediately so caller latches. */
    ex.depth.width = 512; ex.depth.height = 512;
    g_usnap_calls = 0;
    assert(vk_dlss_scene_draw(&ex) == 1);
    assert(!S.have && g_usnap_calls == 0);
    ex.depth.width = 1920; ex.depth.height = 1080;

    /* Eligible target, draw 0 has wrong c3 size: returns 0 so draw 1 retries. */
    ex.ubo[0][0].addr = 0x50000;
    ex.ubo[0][0].size = 64;
    assert(vk_dlss_scene_draw(&ex) == 0);
    assert(!S.have);

    /* Eligible target, right size, but not in usnap: returns 0 so later draw retries. */
    ex.ubo[0][0].size = CAM_BLOCK;
    g_usnap_addr = 0x60000;
    g_usnap_ptr = cam_data;
    assert(vk_dlss_scene_draw(&ex) == 0);
    assert(!S.have && g_usnap_calls == 1);

    /* Draw 2 binds snapshotted CAM_BLOCK: captures scene and returns 1. */
    ex.ubo[0][0].addr = 0x60000;
    assert(vk_dlss_scene_draw(&ex) == 1);
    assert(S.have == 1 && S.serial == 11 && S.cam[0] == 1.0f && S.cam[63] == 64.0f);
    assert(g_tex_locks == 0);

    /* Same pass_serial again: returns 1 immediately without querying usnap. */
    g_usnap_calls = 0;
    assert(vk_dlss_scene_draw(&ex) == 1);
    assert(g_usnap_calls == 0);

    /* Temporal upscaler disabled: returns 1 immediately. */
    state = -1; g_fsr_wanted = 0;
    ex.pass_serial = 12;
    assert(vk_dlss_scene_draw(&ex) == 1);
    assert(S.serial == 11);
}

int main(void) {
    test_pass_snap();
    test_slice_snap_roundtrip_and_shrink();
    test_par_bind_op();
    test_dlss_latch();
    puts("par_collect_test: ok");
    return 0;
}
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', default=shutil.which('clang') or 'clang')
    ap.add_argument('--deps', type=Path, required=True)
    ap.add_argument('--source-root', type=Path, default=ROOT)
    args = ap.parse_args()

    vk_exec_src = (args.source_root / 'runtime/vk_exec.c').read_text()
    vk_dlss_src = (args.source_root / 'runtime/vk_dlss.c').read_text()

    vk_exec_parts = [
        extract_typedef(vk_exec_src, 'ParPassSnap'),
        extract_typedef(vk_exec_src, 'ParSnap'),
        extract_function(vk_exec_src, 'static void par_pass_take('),
        extract_function(vk_exec_src, 'static void par_pass_load('),
        extract_macro(vk_exec_src, 'PAR_SLICE_LOAD'),
        extract_function(vk_exec_src, 'static void par_snap_take('),
        extract_function(vk_exec_src, 'static void par_snap_load('),
        extract_function(vk_exec_src, 'static inline void par_bind_op('),
    ]
    vk_dlss_part = extract_function(vk_dlss_src, 'int vk_dlss_scene_draw(')

    code = (HARNESS
            .replace('/* VK_EXEC_EXTRACT */', '\n\n'.join(vk_exec_parts))
            .replace('/* VK_DLSS_EXTRACT */', vk_dlss_part))

    inc = [
        '-I' + str(args.source_root / 'runtime'),
        '-I' + str(args.deps / 'include'),
        '-I' + str(args.deps / 'include' / 'vma'),
    ]
    with tempfile.TemporaryDirectory(prefix='dkb-par-collect-') as tmp:
        src = Path(tmp) / 'test.c'
        src.write_text(code)
        for opt in ('-O0', '-O2'):
            exe = Path(tmp) / ('par_collect' + opt[2:] + '.exe')
            subprocess.run([args.cc, '-std=gnu11', opt, '-Wall', '-Werror', *inc, str(src), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
