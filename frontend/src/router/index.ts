import { createRouter, createWebHistory } from 'vue-router'
import DataUploadView from '../views/DataUploadView.vue'
import EDAView from '../views/EDAView.vue'
import MLView from '../views/MLView.vue'
import ResultsView from '../views/ResultsView.vue'

const routes = [
  {
    path: '/',
    redirect: '/data'
  },
  {
    path: '/data',
    name: 'DataUpload',
    component: DataUploadView,
    meta: { title: 'データ読み込み' }
  },
  {
    path: '/eda',
    name: 'EDA',
    component: EDAView,
    meta: { title: '探索的データ分析' }
  },
  {
    path: '/ml',
    name: 'ML',
    component: MLView,
    meta: { title: '機械学習設定' }
  },
  {
    path: '/results',
    name: 'Results',
    component: ResultsView,
    meta: { title: '分析結果レポート' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// Update document title
router.beforeEach((to, _, next) => {
  document.title = (to.meta.title as string) || 'ChemAI Nexus'
  next()
})

export default router
