<template>
  <el-container class="main-layout">
    <el-aside width="260px" class="sidebar">
      <div class="brand">
          <div class="logo-box">
              <el-icon><Flask /></el-icon>
          </div>
          <div class="brand-text">
              <h1 class="text-white font-black text-lg leading-tight">ChemAI Nexus</h1>
              <p class="text-[10px] text-sky-400 font-bold tracking-tighter uppercase">Professional Suite v2.0</p>
          </div>
      </div>

      <el-menu 
        router 
        :default-active="activePath"
        class="side-menu"
      >
        <el-menu-item index="/">
            <el-icon><FolderOpened /></el-icon>
            <span>Data Upload</span>
        </el-menu-item>
        <el-menu-item index="/pipeline">
            <el-icon><Setting /></el-icon>
            <span>Pipeline Config</span>
        </el-menu-item>
        <el-menu-item index="/results">
            <el-icon><DataAnalysis /></el-icon>
            <span>Results</span>
        </el-menu-item>
      </el-menu>

      <div class="session-info">
          <p class="text-[9px] text-slate-500 uppercase font-bold mb-1">Active Session</p>
          <p class="text-[10px] text-slate-400 font-mono truncate">{{ sessionId }}</p>
      </div>
    </el-aside>

    <el-main class="content-area">
      <router-view v-slot="{ Component }">
        <transition name="fade-slide" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </el-main>
  </el-container>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue';
import { useRoute } from 'vue-router';
import { Flask, FolderOpened, Setting, DataAnalysis } from '@element-plus/icons-vue';
import { initSession } from './api/client';

const route = useRoute();
const activePath = computed(() => route.path);
const sessionId = ref('');

onMounted(async () => {
    const sid = localStorage.getItem('chemai_session');
    if (!sid) {
        sessionId.value = await initSession();
    } else {
        sessionId.value = sid;
    }
});
</script>

<style>
@import 'element-plus/dist/index.css';

body {
    margin: 0;
    background-color: #020617;
    font-family: 'Inter', -apple-system, sans-serif;
    color: #e2e8f0;
}

.main-layout {
    height: 100vh;
}

.sidebar {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
    display: flex;
    flex-direction: column;
}

.brand {
    padding: 30px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo-box {
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 20px;
}

.side-menu {
    border-right: none !important;
    background-color: transparent !important;
    flex: 1;
}

.el-menu-item {
    color: #94a3b8 !important;
    margin: 4px 12px;
    border-radius: 8px;
    height: 48px !important;
}

.el-menu-item:hover, .el-menu-item.is-active {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
}

.el-menu-item.is-active {
    font-weight: bold;
}

.session-info {
    padding: 20px;
    border-top: 1px solid #1e293b;
    background-color: rgba(15, 23, 42, 0.5);
}

.content-area {
    background: radial-gradient(circle at 50% -20%, #1e293b 0%, #020617 80%);
    padding: 40px !important;
    overflow-y: auto;
}

/* Transitions */
.fade-slide-enter-active, .fade-slide-leave-active {
    transition: all 0.3s ease;
}
.fade-slide-enter-from {
    opacity: 0;
    transform: translateY(10px);
}
.fade-slide-leave-to {
    opacity: 0;
    transform: translateY(-10px);
}
</style>
